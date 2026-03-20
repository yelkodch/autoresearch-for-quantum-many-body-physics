from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare import build_nn_bonds, build_nnn_bonds
from train import NQSModel, fixed_gauge_sign_ratio


class RegressionTests(unittest.TestCase):
    def test_square_lattice_bonds_scale_with_requested_size(self) -> None:
        nn_bonds = build_nn_bonds(6, 6)
        nnn_bonds = build_nnn_bonds(6, 6)

        self.assertEqual(max(max(i, j) for i, j in nn_bonds), 35)
        self.assertEqual(max(max(i, j) for i, j in nnn_bonds), 35)
        self.assertEqual(len(nn_bonds), 72)
        self.assertEqual(len(nnn_bonds), 72)

    def test_log_prob_normalizes_over_fixed_sz_sector(self) -> None:
        model = NQSModel(4, 2)
        configs = torch.tensor(
            [bits for bits in itertools.product((0, 1), repeat=4) if sum(bits) == 2],
            dtype=torch.long,
        )

        with torch.no_grad():
            probs = torch.exp(model.log_prob(configs))

        self.assertAlmostEqual(float(probs.sum().item()), 1.0, places=6)

    def test_checkerboard_gauge_flips_sign_only_across_sublattices(self) -> None:
        configs = torch.zeros((3, 16), dtype=torch.long)

        nn_ratio = fixed_gauge_sign_ratio(configs, 0, 1, mode="checkerboard_marshall")
        nnn_ratio = fixed_gauge_sign_ratio(configs, 0, 5, mode="checkerboard_marshall")

        self.assertTrue(torch.all(nn_ratio == -1.0))
        self.assertTrue(torch.all(nnn_ratio == 1.0))


if __name__ == "__main__":
    unittest.main()
