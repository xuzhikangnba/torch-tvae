import unittest
import torch
from torch.distributions import Gamma, Independent, Normal, StudentT, kl_divergence

from tvae import StudentTPrior


class PriorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(2)

    def test_scalar_density_matches_pytorch(self):
        prior = StudentTPrior(1, df=3.4, loc=1.2, scale=2.3).double()
        z = torch.linspace(-15, 15, 101, dtype=torch.float64).unsqueeze(-1)
        expected = StudentT(prior.df, prior.loc, prior.scale).log_prob(z).squeeze(-1)
        torch.testing.assert_close(prior.log_prob(z), expected)

    def test_posterior_and_gamma_kl(self):
        prior = StudentTPrior(3, df=4.0, scale=[1, 2, 3]).double()
        mean = torch.randn(5, 3, dtype=torch.float64)
        logvar = torch.randn_like(mean)
        posterior = prior.scale_posterior(mean, logvar)
        distance = ((logvar.exp() + mean.square()) / prior.scale.square()).sum(-1)
        torch.testing.assert_close(posterior.mean, (prior.df + 3) / (prior.df + distance))
        expected = kl_divergence(posterior, Gamma(prior.df / 2, prior.df / 2))
        torch.testing.assert_close(prior.terms(mean, logvar).gamma_kl, expected)

    def test_joint_kl_against_independent_monte_carlo(self):
        prior = StudentTPrior(3, df=5.0, loc=[-0.2, 0, 0.4], scale=[1, 1.5, 0.7]).double()
        mean = torch.tensor([[0.3, -0.7, 0.5]], dtype=torch.float64)
        logvar = torch.full_like(mean, -0.8)
        qz = Independent(Normal(mean, (logvar / 2).exp()), 1)
        qu = prior.scale_posterior(mean, logvar)
        pu = Gamma(prior.df / 2, prior.df / 2)
        with torch.no_grad():
            z, u = qz.sample((60000,)), qu.sample((60000,))
            conditional = Independent(Normal(prior.loc, prior.scale / u.sqrt().unsqueeze(-1)), 1)
            joint = qz.log_prob(z) + qu.log_prob(u) - conditional.log_prob(z) - pu.log_prob(u)
            marginal = qz.log_prob(z) - prior.log_prob(z)
            analytic = prior.kl_bound(mean, logvar).item()
            se = joint.std().item() / joint.numel() ** 0.5
            self.assertLess(abs(joint.mean().item() - analytic), 5 * se)
            self.assertGreater(analytic, marginal.mean().item())

    def test_gaussian_limit(self):
        prior = StudentTPrior(3, df=100000.0, scale=[1, 2, 0.7]).double()
        mean = torch.tensor([[0.3, -0.2, 0.5]], dtype=torch.float64)
        logvar = torch.tensor([[-0.5, 0.4, -0.2]], dtype=torch.float64)
        gaussian_kl = kl_divergence(Normal(mean, (logvar / 2).exp()),
                                    Normal(prior.loc, prior.scale)).sum(-1)
        torch.testing.assert_close(prior.kl_bound(mean, logvar), gaussian_kl, atol=1e-4, rtol=1e-4)

    def test_gradients_against_finite_differences(self):
        prior = StudentTPrior(2, learn_loc=True, learn_scale=True).double()
        names, parameters = zip(*prior.named_parameters())
        mean = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        logvar = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        def evaluate(m, lv, *ps):
            return torch.func.functional_call(prior, dict(zip(names, ps)), (m, lv)).sum()
        self.assertTrue(torch.autograd.gradcheck(evaluate, (mean, logvar, *parameters)))

    def test_sampling_scale_and_shared_precision(self):
        prior = StudentTPrior(2, df=8.0, scale=[1, 2], learn_df=False).double()
        with torch.no_grad():
            samples = prior.rsample((100000,))
        expected = prior.scale.square() * prior.df / (prior.df - 2)
        torch.testing.assert_close(samples.var(0), expected, rtol=0.04, atol=0.03)
        # Independent univariate Student-t draws would lose this dependence.
        squared = (samples / prior.scale).square()
        self.assertGreater(torch.corrcoef(squared.T)[0, 1].item(), 0.1)

    def test_shapes_reductions_and_invalid_inputs(self):
        prior = StudentTPrior(3)
        mean = torch.zeros(2, 4, 3)
        value = prior.kl_bound(mean, mean)
        self.assertEqual(value.shape, (2, 4))
        torch.testing.assert_close(prior.kl_bound(mean, mean, "mean"), value.mean())
        torch.testing.assert_close(prior.kl_bound(mean, mean, "sum"), value.sum())
        self.assertEqual(prior.rsample().shape, (3,))
        for kwargs in [dict(df=0), dict(df=float('nan')), dict(scale=0), dict(loc=[1, 2])]:
            with self.assertRaises(ValueError): StudentTPrior(3, **kwargs)
        with self.assertRaises(ValueError): prior.kl_bound(mean, mean, "bad")
        with self.assertRaises(ValueError): prior.kl_bound(mean, torch.zeros(3))

    def test_small_df_has_finite_gradients(self):
        for dtype in [torch.float32, torch.float64]:
            prior = StudentTPrior(4, df=0.01).to(dtype=dtype)
            mean = torch.ones(2, 4, dtype=dtype, requires_grad=True)
            value = prior.kl_bound(mean, torch.zeros_like(mean)).mean()
            value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(mean.grad).all())
            self.assertTrue(torch.isfinite(prior.raw_df.grad))


if __name__ == "__main__": unittest.main()
