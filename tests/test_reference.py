import math
import unittest
import torch
from torch.distributions import Gamma, Independent, Normal
from tvae.reference import TVaDE28, TVaDE128


class ReferenceModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(106)
        torch.set_num_threads(2)

    def test_joint_kl_against_independent_monte_carlo(self):
        model = TVaDE28(3, 3).double()
        with torch.no_grad():
            model.gmm_means.normal_(0, .3)
            model.gmm_logvars.normal_(0, .2)
            model.logit_pi.copy_(torch.tensor([.1, -.3, .2]))
        mu = torch.tensor([[.2, -.4, .1]], dtype=torch.double)
        lv = torch.tensor([[.1, -.2, .3]], dtype=torch.double)
        z0 = mu + .2
        analytic = model.student_t_elbo_loss(z0, mu, lv, eps=0).item()
        nu = model.nu_c.detach()
        loc = model.gmm_means.detach()
        scale = (model.gmm_logvars.detach()/2).exp()
        # Independently evaluate multivariate t densities for the frozen weights.
        q = Independent(Normal(mu[0], (lv[0]/2).exp()), 1)
        radius = ((z0-loc)/scale).square().sum(-1)
        logt = (torch.lgamma((nu+3)/2)-torch.lgamma(nu/2)
                -1.5*torch.log(nu*math.pi)-scale.log().sum(-1)
                -(nu+3)/2*torch.log1p(radius/nu))
        logpi = model.logit_pi.detach().log_softmax(0)
        weights = (logpi+logt).softmax(0)
        n = 40000
        samples = q.sample((n,))
        estimates = torch.zeros(n, dtype=torch.double)
        for c in range(3):
            expected_radius = ((lv[0].exp()+(mu[0]-loc[c]).square())/scale[c].square()).sum()
            qu = Gamma((nu[c]+3)/2, (nu[c]+expected_radius)/2)
            pu = Gamma(nu[c]/2, nu[c]/2)
            u = qu.sample((n,))
            conditional = Independent(Normal(loc[c], scale[c]/u.sqrt()[:,None]), 1)
            joint = q.log_prob(samples)-conditional.log_prob(samples)+qu.log_prob(u)-pu.log_prob(u)
            estimates += weights[c]*(joint+weights[c].log()-logpi[c])
        uncertainty = estimates.std().item()/math.sqrt(n)
        self.assertLess(abs(estimates.mean().item()-analytic), 5*uncertainty+.003)

    def test_multicluster_backward_and_checkpoint_round_trip(self):
        model = TVaDE28(8, 10)
        x = torch.rand(4,1,28,28)
        mu, lv = model.encode(x)
        z = model.reparameterize(mu,lv)
        reconstruction = torch.nn.functional.binary_cross_entropy(model.decode(z),x,reduction='sum')
        loss = reconstruction+5*model.student_t_elbo_loss(z,mu,lv)
        loss.backward()
        for name in ['gmm_means','gmm_logvars','nu_rho','logit_pi']:
            grad = getattr(model,name).grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
        restored = TVaDE28(8,10)
        restored.load_state_dict(model.state_dict(),strict=True)
        model.eval(); restored.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model.encode(x)[0],restored.encode(x)[0]))

    def test_mri_shape_and_loss(self):
        model = TVaDE128(32,2).eval()
        with torch.no_grad():
            mu,lv = model.encode(torch.rand(2,1,128,128))
            self.assertEqual(tuple(model.decode(mu).shape),(2,1,128,128))
            self.assertTrue(torch.isfinite(model.student_t_elbo_loss(mu,mu,lv)))


if __name__ == '__main__':
    unittest.main()
