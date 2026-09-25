"""Finite-state check of the two conditional score identities (no torch needed)."""

import math
import unittest


def poisson(k, rate):
    return math.exp(k * math.log(rate) - rate - math.lgamma(k + 1))


def kl(rate_q, rate_p):
    return rate_q * math.log(rate_q / rate_p) + rate_p - rate_q


def expectation(phi, psi, beta):
    a = math.exp(phi)
    total = 0.
    for k in range(19):
        b = math.exp(psi + .15 * k)
        p1 = 1.6
        p2 = 1. + .2 * k
        for v in range(65):
            q = poisson(k, a) * poisson(v, b)
            reconstruction = (v - 2.) ** 2 / 7.
            total += q * (reconstruction + beta * (kl(a, p1) + kl(b, p2)))
    return total


def conditional_scores(phi, psi, beta, analytic):
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
                signal1 = reconstruction + beta * kl(b, p2)
                signal2 = reconstruction
                direct1 += q * beta * a * math.log(a / p1)
                direct2 += q * beta * b * math.log(b / p2)
            else:
                l1 = math.log(q1 / poisson(k, p1))
                l2 = math.log(q2 / poisson(v, p2))
                signal1 = reconstruction + beta * (l1 + l2)
                signal2 = reconstruction + beta * l2
            # m1*q2*(q1/m1) and q1*m2*(q2/m2): only one-group weights.
            score1 += m1 * q2 * (q1 / m1) * (k - a) * signal1
            score2 += q1 * m2 * (q2 / m2) * (v - b) * signal2
    return score1 + direct1, score2 + direct2


class ConditionalProposalMathTest(unittest.TestCase):
    def test_sampled_and_analytic_kl_match_finite_difference(self):
        phi, psi, beta, step = .2, -.1, .7, 1e-5
        numeric = (
            (expectation(phi + step, psi, beta) - expectation(phi - step, psi, beta)) / (2 * step),
            (expectation(phi, psi + step, beta) - expectation(phi, psi - step, beta)) / (2 * step),
        )
        for analytic in (False, True):
            estimate = conditional_scores(phi, psi, beta, analytic)
            for actual, expected in zip(estimate, numeric):
                self.assertAlmostEqual(actual, expected, delta=2e-4)


if __name__ == '__main__':
    unittest.main()
