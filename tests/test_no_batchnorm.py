import sys
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from model import AutoEncoder
from neural_operations import Conv2D
from test_mixed_model import _args
from train_poisson_posterior_rate import load_checkpoint
import utils


@pytest.mark.parametrize('rate_only', [False, True])
@pytest.mark.parametrize('explicit_arch', [None, 'res_mbconv'])
def test_training_entry_points_select_architecture(tmp_path, rate_only, explicit_arch):
    import train

    argv = ['train.py', '--root', str(tmp_path), '--save', 'cli',
            '--latent_distribution', 'poisson']
    if explicit_arch is not None:
        argv.extend(['--arch_instance', explicit_arch])
    with patch.object(sys, 'argv', argv), patch.object(train, 'init_processes') as launch:
        train.main_cli(posterior_rate_only=rate_only)
    args = launch.call_args[0][-1]
    assert args.arch_instance == (explicit_arch or 'res_elu')
    assert getattr(args, 'posterior_rate_only', False) == rate_only


@pytest.fixture
def single_rank_collectives():
    # Keep the real ELUConv data initialization; only emulate the single-rank
    # transport so CPU tests do not need a listening socket or NCCL.
    with patch.object(dist, 'get_world_size', return_value=1), \
            patch.object(dist, 'all_reduce') as reduce:
        yield reduce


@pytest.mark.parametrize('bias', [False, True])
def test_data_init_preserves_fp32_parameters_with_fp16_output(single_rank_collectives, bias):
    torch.manual_seed(31)
    layer = Conv2D(2, 3, 3, padding=1, bias=bias, data_init=True)
    parameters = dict(layer.named_parameters())
    optimizer = torch.optim.Adamax(layer.parameters(), lr=0.001)
    conv2d = F.conv2d

    def reduced_precision_conv(*args, **kwargs):
        # Emulate autocast's FP16 convolution output on CPU. The initialization
        # must not replace FP32 parameter storage with that output's dtype.
        result = conv2d(*args, **kwargs)
        return result.half() if not layer.init_done else result

    with patch('neural_operations.F.conv2d', side_effect=reduced_precision_conv):
        output = layer(torch.randn(4, 2, 8, 8))
    assert layer.init_done
    for name, parameter in layer.named_parameters():
        assert parameter is parameters[name]
        assert parameter.dtype == torch.float32, name
        assert torch.isfinite(parameter).all(), name
    assert single_rank_collectives.call_count == 2
    for call in single_rank_collectives.call_args_list:
        assert call[0][0].dtype == torch.float32
    output.square().mean().backward()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all()
    before = layer.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(layer.weight, before)


@pytest.mark.parametrize('hierarchical', [False, True])
@pytest.mark.parametrize('image_likelihood', ['dataset_default', 'gaussian_fixed'])
def test_bn_free_forward_backward_and_checkpoint(single_rank_collectives, tmp_path, hierarchical,
                                                image_likelihood):
    torch.manual_seed(7)
    args = _args()
    args.arch_instance = 'res_elu'
    args.latent_distribution = 'poisson'
    args.image_likelihood = image_likelihood
    args.gaussian_variance = 0.01
    args.num_latent_scales = 2 if hierarchical else 1
    args.num_groups_per_scale = 2 if hierarchical else 1
    args.ada_groups = False
    args.num_preprocess_blocks = args.num_postprocess_blocks = 2 if hierarchical else 3
    args.num_preprocess_cells = args.num_postprocess_cells = 2
    model = AutoEncoder(args, None, utils.get_arch_cells(args.arch_instance)).train()

    assert not any(isinstance(module, _BatchNorm) for module in model.modules())
    assert model.all_bn_layers == []
    assert model.batchnorm_loss() == 0
    assert not any(name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))
                   for name in model.state_dict())
    initialized_convs = [module for module in model.modules()
                         if isinstance(module, Conv2D) and module.data_init]
    assert initialized_convs and all(not module.init_done for module in initialized_convs)

    x = torch.rand(2, 1, 32, 32)
    if image_likelihood == 'dataset_default':
        x = x.round()
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)
    logits, log_q, log_p, kl_all, _ = model(x)
    assert logits.shape == x.shape
    assert all(module.init_done for module in initialized_convs)
    assert single_rank_collectives.call_count == 2 * len(initialized_convs)
    assert all(torch.isfinite(value).all() for value in [logits, log_q, log_p] + kl_all)
    recon = utils.reconstruction_loss(model.decoder_output(logits), x, crop=model.crop_output)
    loss = (recon + sum(kl_all)).mean() + model.batchnorm_loss()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and (p.grad != 0).any()
               for p in model.enc_sampler.parameters())
    if not hierarchical:
        assert model.prior_log_rate.grad is not None
        before = model.prior_log_rate.detach().clone()
    optimizer.step()
    if not hierarchical:
        assert not torch.equal(before, model.prior_log_rate)

    path = tmp_path / 'checkpoint.pt'
    torch.save({'args': args, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, path)
    checkpoint = load_checkpoint(path)
    saved_args = checkpoint['args']
    restored = AutoEncoder(saved_args, None, utils.get_arch_cells(saved_args.arch_instance))
    restored.load_state_dict(checkpoint['state_dict'], strict=True)
    restored_optimizer = torch.optim.Adamax(restored.parameters(), lr=0.001)
    restored_optimizer.load_state_dict(checkpoint['optimizer'])

    # Both resume and evaluation must preserve initialized checkpoint parameters.
    # Do not manually set init_done here: loading the checkpoint must handle it.
    for training in (True, False):
        model.train(training)
        restored.train(training)
        with torch.no_grad():
            torch.manual_seed(19)
            expected = model(1. - x)
            torch.manual_seed(19)
            actual = restored(1. - x)
        for index in range(3):
            assert torch.equal(expected[index], actual[index])
        for name, value in restored.state_dict().items():
            assert torch.equal(value, checkpoint['state_dict'][name]), name

    with torch.no_grad():
        torch.manual_seed(23)
        expected = model.sample(2, t=1.)
        torch.manual_seed(23)
        actual = restored.sample(2, t=1.)
    assert torch.equal(expected, actual)
    assert torch.isfinite(actual).all()
    assert single_rank_collectives.call_count == 2 * len(initialized_convs)
