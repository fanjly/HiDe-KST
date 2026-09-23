import unittest
import sys
import types
from pathlib import Path

import torch


# Import only HiDe-KST for this unit test. The top-level models package also
# imports every optional baseline and would make this focused test depend on
# their unrelated third-party packages.
if 'models' not in sys.modules:
    models_package = types.ModuleType('models')
    models_package.__path__ = [str(Path(__file__).parents[1] / 'models')]
    sys.modules['models'] = models_package

from models.hkst.layers import AdaptiveSpatialFactorization
from models.hkst.model import HKST


class DynamicGraphBasisTest(unittest.TestCase):
    def test_static_path_matches_original_formula(self):
        torch.manual_seed(5)
        layer = AdaptiveSpatialFactorization(
            num_nodes=5,
            rank=3,
            input_dim=4,
            output_dim=4,
            hidden_dim=8,
            hidden_layers=2,
            res=False,
            dropout=0.0,
            adj_matrix=torch.eye(5),
        )
        layer.eval()
        x = torch.randn(2, 5, 3, 4)

        actual = layer(x)
        old_gso = torch.relu(layer.P.t() @ (layer.adj @ layer.P))
        expected = torch.einsum('bnfc,nr->brfc', x, layer.P)
        expected = layer.gcn(expected, old_gso)
        expected = torch.einsum('brfc,rn->bnfc', expected, layer.P.t())

        torch.testing.assert_close(actual, expected)

    def test_dynamic_factorization_forward_and_backward(self):
        torch.manual_seed(7)
        layer = AdaptiveSpatialFactorization(
            num_nodes=5,
            rank=3,
            input_dim=4,
            output_dim=4,
            hidden_dim=8,
            hidden_layers=2,
            res=False,
            dropout=0.0,
            adj_matrix=torch.eye(5),
            use_dynamic_graph_basis=True,
            num_dynamic_graph_bases=4,
            dynamic_graph_scale=0.1,
        )
        x = torch.randn(2, 5, 3, 4, requires_grad=True)
        output = layer(x)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(layer._last_router_weights.shape, (2, 4))
        torch.testing.assert_close(
            layer._last_router_weights.sum(dim=-1), torch.ones(2)
        )

        loss = output.square().mean() + layer.get_dynamic_graph_balance_loss()
        loss.backward()
        self.assertIsNotNone(layer.graph_bases.grad)
        self.assertIsNotNone(layer.graph_router[-1].weight.grad)

    def test_hkst_dynamic_graph_end_to_end(self):
        torch.manual_seed(11)
        model = HKST(
            input_len=8,
            output_len=4,
            num_nodes=5,
            num_feat=1,
            hidden_dim=8,
            spat_layers=2,
            temp_layers=2,
            spat_rank=3,
            temp_rank=4,
            spat_decomp=True,
            use_node_modem=True,
            use_graph=True,
            patch_sizes=[4, 6],
            strides=[2, 2],
            dropout=0.0,
            recon_mask_ratio=0.5,
            use_timestamp=False,
            adj_matrix=torch.eye(5),
            use_dynamic_graph_basis=True,
            num_dynamic_graph_bases=2,
            dynamic_graph_scale=0.1,
            dynamic_graph_balance_weight=0.01,
        )
        x = torch.randn(2, 8, 5)
        output = model(x, None, None)
        loss = output.square().mean() + model.get_dynamic_graph_loss()
        loss.backward()

        self.assertEqual(output.shape, (2, 4, 5))
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
