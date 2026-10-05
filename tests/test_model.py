import io
import unittest
from unittest.mock import patch

import torch
from torch import nn
from tvae import TVAE


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_label_free_training_and_sampling(self):
        model = TVAE(input_dim=12, latent_dim=3, hidden_dims=(16,))
        x = torch.rand(8, 1, 3, 4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(3):
            opt.zero_grad()
            result = model.loss(x)
            result.loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            opt.step()
        self.assertEqual(model.reconstruct(x).shape, x.shape)
        self.assertEqual(model.sample(5).shape, (5, 12))
        self.assertEqual(result.per_sample_loss.shape, (8,))

    def test_batch_replication_preserves_loss_and_gradient(self):
        model = TVAE(input_dim=5, latent_dim=2, hidden_dims=(8,)).double()
        x = torch.rand(3, 5, dtype=torch.float64)
        # Fix the reconstruction draw to compare equivalent replicated batches.
        with patch('torch.randn_like', side_effect=lambda x: torch.zeros_like(x)):
            a = model.loss(x, beta=0.7)
            ga = torch.autograd.grad(a.loss, tuple(model.parameters()))
            b = model.loss(x.repeat(4, 1), beta=0.7)
            gb = torch.autograd.grad(b.loss, tuple(model.parameters()))
        torch.testing.assert_close(a.loss, b.loss)
        for first, second in zip(ga, gb): torch.testing.assert_close(first, second)

    def test_state_dict_roundtrip(self):
        model = TVAE(input_dim=5, latent_dim=2, df=3.0, learn_df=False).double()
        x = torch.randn(3, 5, dtype=torch.float64)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = TVAE(input_dim=5, latent_dim=2, df=3.0, learn_df=False).double()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(model.reconstruct(x), restored.reconstruct(x))

    def test_custom_backbone_and_gaussian_likelihood(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(6, 4)
            def forward(self, x): return self.fc(x).chunk(2, -1)
        model = TVAE(latent_dim=2, encoder=Encoder(), decoder=nn.Linear(2, 6),
                     likelihood='gaussian', observation_scale=0.5)
        x = torch.randn(4, 6)
        result = model.loss(x)
        result.loss.backward()
        self.assertTrue(torch.isfinite(result.loss))
        self.assertEqual(model.sample(3).shape, (3, 6))

    def test_input_contracts(self):
        with self.assertRaises(ValueError): TVAE()
        with self.assertRaises(ValueError): TVAE(input_dim=3, likelihood='unknown')
        with self.assertRaises(ValueError): TVAE(encoder=nn.Identity())
        model = TVAE(input_dim=3)
        with self.assertRaises(ValueError): model.loss(torch.zeros(0, 3))
        with self.assertRaises(ValueError): model.loss(torch.full((2, 3), -1.0))
        with self.assertRaises(ValueError): model.loss(torch.zeros(2, 3), beta=-1)


if __name__ == '__main__': unittest.main()
