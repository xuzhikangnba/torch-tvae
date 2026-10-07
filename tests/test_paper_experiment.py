import argparse
import unittest

import numpy as np
import torch

from tvae.paper_experiment import add_cauchy, metrics, objective, optimizer_for, initialize_posterior_variance
from tvae.reference import TVaDE28


class PaperExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)

    def test_loss_uses_same_per_observation_reduction(self):
        model = TVaDE28(8, 10).eval()
        # Fix the sampled latent for this invariance test only.
        model.reparameterize = lambda mu, lv: mu
        x = torch.rand(2, 1, 28, 28)
        single = objective(model, x)
        repeated = objective(model, x.repeat(3, 1, 1, 1))
        for a, b in zip(single, repeated):
            torch.testing.assert_close(a, b, atol=.001, rtol=1e-5)

    def test_adam_groups_cover_parameters_once_and_use_stated_rates(self):
        model = TVaDE28(8, 10)
        args = argparse.Namespace(ae_lr=.001, mixture_lr=.002, optimizer="adam", weight_decay=0.)
        optimizer = optimizer_for(model, args)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [.001, .002])
        pretrain = optimizer_for(model, args, pretrain=True)
        self.assertNotIn(id(model.gmm_means), [id(p) for g in pretrain.param_groups for p in g["params"]])

    def test_eval_noise_repeats_independently_of_training_rng(self):
        image = torch.full((8, 1, 28, 28), .5)
        a = add_cauchy(image, .12, torch.Generator().manual_seed(123))
        torch.rand(200)
        b = add_cauchy(image, .12, torch.Generator().manual_seed(123))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(add_cauchy(image, 0), image))
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue((a >= 0).all() and (a <= 1).all())

    def test_no_dummy_medical_accuracy_and_undefined_collapse_metrics(self):
        result = metrics(np.arange(12).reshape(6, 2), np.full(6, -1), np.zeros(6, dtype=int), 2)
        self.assertNotIn("accuracy", result)
        self.assertIsNone(result["silhouette"])
        self.assertIsNone(result["davies_bouldin"])
        self.assertEqual(result["occupied_clusters"], 1)

    def test_ae_to_vae_initialization_preserves_means_and_sets_unit_variance(self):
        model = TVaDE28(8, 10).eval()
        x = torch.rand(2, 1, 28, 28)
        with torch.no_grad():
            mean_before = model.encode(x)[0].clone()
            model.encoder.fc_logvar.weight.fill_(100)
            initialize_posterior_variance(model, "unit")
            mu, lv = model.encode(x)
        torch.testing.assert_close(mu, mean_before)
        self.assertTrue(torch.equal(lv, torch.zeros_like(lv)))


if __name__ == "__main__":
    unittest.main()
