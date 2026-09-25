"""Finite-state check of the two conditional score identities (no torch needed)."""

import math
import unittest


def poisson(k, rate):
    return math.exp(k * math.log(rate) - rate - math.lgamma(k + 1))


def kl(rate_q, rate_p):
    return rate_q * math.log(rate_q / rate_p) + rate_p - rate_q


def expectation(phi, psi, beta, coefficients):
    a = math.exp(phi)
    total = 0.
    for k in range(19):
        b = math.exp(psi + .15 * k)
        p1 = 1.6
        p2 = 1. + .2 * k
        for v in range(65):
            q = poisson(k, a) * poisson(v, b)
            reconstruction = (v - 2.) ** 2 / 7.
            total += q * (reconstruction + beta * (
                coefficients[0] * kl(a, p1) + coefficients[1] * kl(b, p2)))
    return total


def conditional_scores(phi, psi, beta, analytic, coefficients):
    a = math.exp(phi)
    score1 = score2 = direct1 = direct2 = 0.
    for k in range(19):
        b = math.exp(psi + .15 * k)
        p1 = 1.6
        p2 = 1. + .2 * k
        q1 = poisson(k, a)
        m1 = (q1 + poisson(k, math.sqrt(a))) / 2.
        if not m1:
            continue
        for v in range(65):
            q2 = poisson(v, b)
            m2 = (q2 + poisson(v, math.sqrt(b))) / 2.
            if not m2:
                continue
            reconstruction = (v - 2.) ** 2 / 7.
            q = q1 * q2
            if analytic:
                signal1 = reconstruction + beta * coefficients[1] * kl(b, p2)
                signal2 = reconstruction
                direct1 += q * beta * coefficients[0] * a * math.log(a / p1)
                direct2 += q * beta * coefficients[1] * b * math.log(b / p2)
            else:
                l1 = math.log(q1 / poisson(k, p1))
                l2 = math.log(q2 / poisson(v, p2))
                signal1 = reconstruction + beta * (
                    coefficients[0] * l1 + coefficients[1] * l2)
                signal2 = reconstruction + beta * coefficients[1] * l2
            # m1*q2*(q1/m1) and q1*m2*(q2/m2): only one-group weights.
            score1 += m1 * q2 * (q1 / m1) * (k - a) * signal1
            score2 += q1 * m2 * (q2 / m2) * (v - b) * signal2
    return score1 + direct1, score2 + direct2


class ConditionalProposalMathTest(unittest.TestCase):
    def test_poisson_conditional_dmis_baseline_preserves_mean_and_reduces_variance(self):
        rate = 2.
        proposal_rate = rate ** (1. / 3.)
        terms = []
        for k in range(35):
            q = poisson(k, rate)
            m = (q + poisson(k, proposal_rate)) / 2.
            weight = q / m
            score = k - rate
            signal = (k - 2.) ** 2 + 1.
            terms.append((m, weight, score, signal))
        numerator = sum(m * w ** 2 * h ** 2 * f for m, w, h, f in terms)
        denominator = sum(m * w ** 2 * h ** 2 for m, w, h, f in terms)
        baseline = numerator / denominator

        def moments(b):
            mean = sum(m * w * h * (f - b) for m, w, h, f in terms)
            second = sum(m * (w * h * (f - b)) ** 2 for m, w, h, f in terms)
            return mean, second - mean ** 2

        mean_zero, var_zero = moments(0.)
        mean_opt, var_opt = moments(baseline)
        self.assertAlmostEqual(mean_zero, mean_opt, delta=1e-10)
        self.assertLess(var_opt, var_zero)

    def test_sampled_and_analytic_kl_match_finite_difference(self):
        phi, psi, beta, step = .2, -.1, .7, 1e-5
        for coefficients in ((1., 1.), (1.7, .6)):
            numeric = (
                (expectation(phi + step, psi, beta, coefficients) -
                 expectation(phi - step, psi, beta, coefficients)) / (2 * step),
                (expectation(phi, psi + step, beta, coefficients) -
                 expectation(phi, psi - step, beta, coefficients)) / (2 * step),
            )
            for analytic in (False, True):
                estimate = conditional_scores(phi, psi, beta, analytic, coefficients)
                for actual, expected in zip(estimate, numeric):
                    self.assertAlmostEqual(actual, expected, delta=2e-4)


if __name__ == '__main__':
    unittest.main()
