from types import SimpleNamespace
import unittest

import torch

from model import AutoEncoder
import utils


def _args(dual_decoder_paths, res_dist=True):
    return SimpleNamespace(
        dataset='mnist',
        use_se=False,
        res_dist=res_dist,
        dual_decoder_paths=dual_decoder_paths,
        num_x_bits=8,
        num_latent_scales=2,
        num_groups_per_scale=1,
        num_latent_per_group=2,
        ada_groups=False,
        min_groups_per_scale=1,
        num_channels_enc=4,
        num_channels_dec=4,
        num_preprocess_blocks=0,
        num_preprocess_cells=1,
        num_cell_per_cond_enc=1,
        num_postprocess_blocks=0,
        num_postprocess_cells=1,
        num_cell_per_cond_dec=1,
        num_mixture_dec=1,
        num_nf=0,
    )


def _model(dual_decoder_paths, res_dist=True):
    model = AutoEncoder(_args(dual_decoder_paths, res_dist), None, utils.get_arch_cells('res_elu'))
    # The legacy data-dependent initialization assumes an initialized distributed
    # process group. It is unrelated to these topology tests.
    for module in model.modules():
        if hasattr(module, 'init_done'):
            module.init_done = True
    return model


def _count_calls(modules):
    counts = {id(module): 0 for module in modules}
    hooks = []

    for module in modules:
        def increment(_, __, ___, key=id(module)):
            counts[key] += 1

        hooks.append(module.register_forward_hook(increment))

    return counts, hooks


class DualDecoderPathsTest(unittest.TestCase):
    def test_missing_flag_keeps_old_checkpoint_args_compatible(self):
        args = _args(dual_decoder_paths=False)
        del args.dual_decoder_paths
        model = AutoEncoder(args, None, utils.get_arch_cells('res_elu'))
        self.assertFalse(model.dual_decoder_paths)

    def test_dual_path_requires_residual_posterior(self):
        with self.assertRaisesRegex(ValueError, 'requires --res_dist'):
            _model(dual_decoder_paths=True, res_dist=False)

    def test_forward_reuses_ordinary_cells_but_not_combiners(self):
        model = _model(dual_decoder_paths=True)
        ordinary_cells = [cell for cell in model.dec_tower if cell.cell_type != 'combiner_dec']
        prior_combiners = [cell for cell in model.dec_tower if cell.cell_type == 'combiner_dec']
        posterior_combiners = list(model.posterior_dec_combiners)

        self.assertEqual(len(prior_combiners), sum(model.groups_per_scale))
        self.assertEqual(len(posterior_combiners), len(prior_combiners))
        for prior, posterior in zip(prior_combiners, posterior_combiners):
            self.assertNotEqual(id(prior), id(posterior))
            self.assertTrue(set(prior.parameters()).isdisjoint(set(posterior.parameters())))

        all_modules = ordinary_cells + prior_combiners + posterior_combiners
        counts, hooks = _count_calls(all_modules)
        try:
            with torch.no_grad():
                logits, _, _, kl_all, kl_diag = model(torch.rand(2, 1, 32, 32))
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(tuple(logits.shape), (2, 1, 32, 32))
        self.assertEqual(len(kl_all), sum(model.groups_per_scale))
        self.assertEqual(len(kl_diag), sum(model.groups_per_scale))
        for cell in ordinary_cells:
            self.assertEqual(counts[id(cell)], 2)
        for cell in prior_combiners + posterior_combiners:
            self.assertEqual(counts[id(cell)], 1)

    def test_default_path_calls_each_decoder_cell_once(self):
        model = _model(dual_decoder_paths=False)
        self.assertEqual(len(model.posterior_dec_combiners), 0)
        cells = list(model.dec_tower)
        counts, hooks = _count_calls(cells)
        try:
            with torch.no_grad():
                model(torch.rand(2, 1, 32, 32))
        finally:
            for hook in hooks:
                hook.remove()

        for cell in cells:
            self.assertEqual(counts[id(cell)], 1)

    def test_sampling_uses_only_the_prior_path(self):
        model = _model(dual_decoder_paths=True)
        prior_combiners = [cell for cell in model.dec_tower if cell.cell_type == 'combiner_dec']
        posterior_combiners = list(model.posterior_dec_combiners)
        counts, hooks = _count_calls(prior_combiners + posterior_combiners)
        try:
            with torch.no_grad():
                logits = model.sample(num_samples=2, t=1.0)
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(tuple(logits.shape), (2, 1, 32, 32))
        for cell in prior_combiners:
            self.assertEqual(counts[id(cell)], 1)
        for cell in posterior_combiners:
            self.assertEqual(counts[id(cell)], 0)


if __name__ == '__main__':
    unittest.main()
