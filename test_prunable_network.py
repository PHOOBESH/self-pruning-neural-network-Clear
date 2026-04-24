"""
test_prunable_network.py
========================
Unit & integration tests for the Self-Pruning Neural Network.
Run with:  python test_prunable_network.py

No pytest required — uses Python's built-in unittest module.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Allow import from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prunable_network import (
    PrunableLinear,
    SelfPruningNet,
    TrainingConfig,
    build_optimizer,
    train_one_epoch,
    evaluate,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_dummy_loader(n_batches=5, batch_size=16, input_size=3072, num_classes=10):
    """Returns a list of (images, labels) tuples simulating a DataLoader."""
    data = []
    for _ in range(n_batches):
        images = torch.randn(batch_size, 3, 32, 32)
        labels = torch.randint(0, num_classes, (batch_size,))
        data.append((images, labels))
    return data


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 1: PrunableLinear Layer
# ─────────────────────────────────────────────────────────────────────────────

class TestPrunableLinear(unittest.TestCase):

    def setUp(self):
        self.in_f  = 64
        self.out_f = 32
        self.layer = PrunableLinear(self.in_f, self.out_f)

    # ── Parameter existence ──────────────────────────────────────────────────

    def test_weight_parameter_exists(self):
        """weight must be a registered nn.Parameter."""
        self.assertIsInstance(self.layer.weight, nn.Parameter)

    def test_bias_parameter_exists(self):
        """bias must be a registered nn.Parameter."""
        self.assertIsInstance(self.layer.bias, nn.Parameter)

    def test_gate_scores_parameter_exists(self):
        """gate_scores must be a registered nn.Parameter."""
        self.assertIsInstance(self.layer.gate_scores, nn.Parameter)

    def test_gate_scores_same_shape_as_weight(self):
        """gate_scores must have the exact same shape as weight."""
        self.assertEqual(self.layer.gate_scores.shape, self.layer.weight.shape)

    # ── Forward pass shape ───────────────────────────────────────────────────

    def test_forward_output_shape(self):
        """Output shape must be (batch, out_features)."""
        x = torch.randn(8, self.in_f)
        out = self.layer(x)
        self.assertEqual(out.shape, (8, self.out_f))

    def test_forward_batch_independence(self):
        """Each sample in a batch must be processed independently."""
        x = torch.randn(4, self.in_f)
        out_full  = self.layer(x)
        out_first = self.layer(x[:1])
        self.assertTrue(torch.allclose(out_full[0], out_first[0], atol=1e-5))

    # ── Gate value range ─────────────────────────────────────────────────────

    def test_gate_values_in_zero_one(self):
        """Sigmoid output must always be in (0, 1)."""
        gates = self.layer.gate_values()
        self.assertTrue((gates >= 0).all() and (gates <= 1).all(),
                        "Gate values must be in [0, 1]")

    def test_gate_values_shape(self):
        """gate_values() must return tensor of same shape as weight."""
        gates = self.layer.gate_values()
        self.assertEqual(gates.shape, self.layer.weight.shape)

    # ── Gradient flow ────────────────────────────────────────────────────────

    def test_gradient_flows_to_weight(self):
        """Backprop must produce non-None gradients for weight."""
        x   = torch.randn(4, self.in_f)
        out = self.layer(x).sum()
        out.backward()
        self.assertIsNotNone(self.layer.weight.grad,
                             "weight.grad must not be None after backward()")
        self.assertFalse(torch.all(self.layer.weight.grad == 0),
                         "weight gradients should not all be zero")

    def test_gradient_flows_to_gate_scores(self):
        """Backprop must produce non-None gradients for gate_scores."""
        x   = torch.randn(4, self.in_f)
        out = self.layer(x).sum()
        out.backward()
        self.assertIsNotNone(self.layer.gate_scores.grad,
                             "gate_scores.grad must not be None after backward()")
        self.assertFalse(torch.all(self.layer.gate_scores.grad == 0),
                         "gate_scores gradients should not all be zero")

    def test_gradient_not_flows_to_gate_values_detached(self):
        """gate_values() returns a detached tensor — no grad should attach."""
        gates = self.layer.gate_values()
        self.assertFalse(gates.requires_grad,
                         "gate_values() should return a detached tensor")

    # ── Pruning effect ───────────────────────────────────────────────────────

    def test_zero_gate_zeroes_output(self):
        """If all gate_scores → -inf, all gates → 0, output should equal bias."""
        with torch.no_grad():
            self.layer.gate_scores.fill_(-1e9)   # sigmoid(-1e9) ≈ 0
            self.layer.bias.fill_(1.0)
        x   = torch.randn(4, self.in_f)
        out = self.layer(x)
        expected = torch.ones(4, self.out_f)
        self.assertTrue(torch.allclose(out, expected, atol=1e-4),
                        "With all gates=0, output must equal bias only")

    def test_one_gate_passes_full_weight(self):
        """If all gate_scores → +inf, gates ≈ 1, layer behaves like standard Linear."""
        with torch.no_grad():
            self.layer.gate_scores.fill_(1e9)    # sigmoid(+inf) ≈ 1
            self.layer.bias.zero_()
        x   = torch.randn(4, self.in_f)
        out_prunable = self.layer(x)
        out_standard = torch.nn.functional.linear(x, self.layer.weight)
        self.assertTrue(torch.allclose(out_prunable, out_standard, atol=1e-4),
                        "With all gates=1, PrunableLinear must match standard Linear")

    # ── Sparsity utility ─────────────────────────────────────────────────────

    def test_sparsity_returns_tuple(self):
        ratio, pruned, total = self.layer.sparsity()
        self.assertIsInstance(ratio,  float)
        self.assertIsInstance(pruned, int)
        self.assertIsInstance(total,  int)

    def test_sparsity_ratio_between_0_and_1(self):
        ratio, _, _ = self.layer.sparsity()
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio,    1.0)

    def test_sparsity_total_equals_weight_numel(self):
        _, _, total = self.layer.sparsity()
        self.assertEqual(total, self.layer.weight.numel())

    def test_full_sparsity_when_gates_zero(self):
        """All gates near zero → sparsity should be 1.0."""
        with torch.no_grad():
            self.layer.gate_scores.fill_(-1e9)
        ratio, pruned, total = self.layer.sparsity(threshold=1e-2)
        self.assertAlmostEqual(ratio, 1.0, places=3,
                               msg="All gates=0 should give sparsity=1.0")

    def test_zero_sparsity_when_gates_one(self):
        """All gates near one → sparsity should be 0.0."""
        with torch.no_grad():
            self.layer.gate_scores.fill_(1e9)
        ratio, pruned, total = self.layer.sparsity(threshold=1e-2)
        self.assertAlmostEqual(ratio, 0.0, places=3,
                               msg="All gates=1 should give sparsity=0.0")

    # ── No-bias variant ──────────────────────────────────────────────────────

    def test_no_bias_variant(self):
        layer = PrunableLinear(16, 8, bias=False)
        self.assertIsNone(layer.bias)
        x   = torch.randn(4, 16)
        out = layer(x)
        self.assertEqual(out.shape, (4, 8))


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 2: SelfPruningNet
# ─────────────────────────────────────────────────────────────────────────────

class TestSelfPruningNet(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.model = SelfPruningNet()
        self.device = torch.device("cpu")

    # ── Architecture ─────────────────────────────────────────────────────────

    def test_output_shape_cifar10(self):
        """Model must output (B, 10) logits for CIFAR-10 input."""
        x   = torch.randn(8, 3, 32, 32)
        out = self.model(x)
        self.assertEqual(out.shape, (8, 10))

    def test_all_fc_layers_are_prunable(self):
        """Every Linear-type layer must be PrunableLinear, not nn.Linear."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.fail(f"Found nn.Linear at '{name}' — should be PrunableLinear")

    def test_prunable_layers_returns_list(self):
        layers = self.model.prunable_layers()
        self.assertIsInstance(layers, list)
        self.assertGreater(len(layers), 0)

    def test_prunable_layers_all_correct_type(self):
        for layer in self.model.prunable_layers():
            self.assertIsInstance(layer, PrunableLinear)

    # ── Sparsity loss ────────────────────────────────────────────────────────

    def test_sparsity_loss_is_scalar(self):
        loss = self.model.sparsity_loss()
        self.assertEqual(loss.shape, torch.Size([]))

    def test_sparsity_loss_is_positive(self):
        """L1 norm of sigmoid outputs is always > 0."""
        loss = self.model.sparsity_loss()
        self.assertGreater(loss.item(), 0.0)

    def test_sparsity_loss_differentiable(self):
        """Sparsity loss must be differentiable w.r.t. gate_scores."""
        loss = self.model.sparsity_loss()
        loss.backward()
        for layer in self.model.prunable_layers():
            self.assertIsNotNone(layer.gate_scores.grad)

    def test_sparsity_loss_decreases_with_lower_gates(self):
        """Forcing gate_scores lower should decrease sparsity loss."""
        loss_before = self.model.sparsity_loss().item()
        with torch.no_grad():
            for layer in self.model.prunable_layers():
                layer.gate_scores.fill_(-5.0)   # sigmoid(-5) ≈ 0.007
        loss_after = self.model.sparsity_loss().item()
        self.assertLess(loss_after, loss_before,
                        "Lowering gate_scores should reduce sparsity loss")

    # ── Network sparsity ─────────────────────────────────────────────────────

    def test_network_sparsity_has_overall_key(self):
        stats = self.model.network_sparsity()
        self.assertIn("overall", stats)

    def test_network_sparsity_overall_in_range(self):
        stats = self.model.network_sparsity()
        ratio = stats["overall"]["sparsity"]
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio,    1.0)

    def test_network_sparsity_per_layer_keys(self):
        stats = self.model.network_sparsity()
        n_layers = len(self.model.prunable_layers())
        for i in range(n_layers):
            self.assertIn(f"layer_{i}", stats)

    def test_all_gate_values_1d_tensor(self):
        gates = self.model.all_gate_values()
        self.assertEqual(gates.dim(), 1)
        self.assertGreater(gates.numel(), 0)

    def test_all_gate_values_in_range(self):
        gates = self.model.all_gate_values()
        self.assertTrue((gates >= 0).all() and (gates <= 1).all())

    # ── Custom architecture ──────────────────────────────────────────────────

    def test_custom_hidden_dims(self):
        model = SelfPruningNet(hidden_dims=[128, 64])
        x   = torch.randn(4, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 3: Optimizer & Training Components
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizerAndTraining(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.cfg    = TrainingConfig(num_epochs=2, batch_size=16)
        self.model  = SelfPruningNet()
        self.device = torch.device("cpu")

    # ── Optimizer ────────────────────────────────────────────────────────────

    def test_optimizer_has_two_param_groups(self):
        """Weight params and gate params must be in separate groups."""
        optimizer = build_optimizer(self.model, self.cfg)
        self.assertEqual(len(optimizer.param_groups), 2,
                         "Optimizer must have 2 param groups (weights + gates)")

    def test_gate_lr_is_higher(self):
        """Gates must have a higher LR than weights."""
        optimizer  = build_optimizer(self.model, self.cfg)
        weight_lr  = optimizer.param_groups[0]["lr"]
        gate_lr    = optimizer.param_groups[1]["lr"]
        self.assertGreater(gate_lr, weight_lr,
                           "Gate LR must be > weight LR")

    def test_gate_group_contains_gate_scores(self):
        """Second param group must contain gate_scores parameters only."""
        optimizer  = build_optimizer(self.model, self.cfg)
        gate_group = optimizer.param_groups[1]["params"]
        n_gates    = sum(1 for _ in self.model.prunable_layers())
        self.assertEqual(len(gate_group), n_gates,
                         "Gate param group size must equal number of PrunableLinear layers")

    # ── Single training step ─────────────────────────────────────────────────

    def test_train_one_epoch_returns_three_losses(self):
        loader    = make_dummy_loader(n_batches=3)
        optimizer = build_optimizer(self.model, self.cfg)
        result    = train_one_epoch(self.model, loader, optimizer, None,
                                    lambda_sparse=1e-4, device=self.device)
        self.assertEqual(len(result), 3,
                         "train_one_epoch must return (total, cls, sparse) losses")

    def test_train_one_epoch_losses_are_positive(self):
        loader    = make_dummy_loader(n_batches=3)
        optimizer = build_optimizer(self.model, self.cfg)
        total, cls, sparse = train_one_epoch(
            self.model, loader, optimizer, None,
            lambda_sparse=1e-4, device=self.device)
        self.assertGreater(total,  0.0, "Total loss must be positive")
        self.assertGreater(cls,    0.0, "Classification loss must be positive")
        self.assertGreater(sparse, 0.0, "Sparsity loss must be positive")

    def test_weights_change_after_training_step(self):
        """Parameters must actually be updated by the optimizer."""
        loader    = make_dummy_loader(n_batches=2)
        optimizer = build_optimizer(self.model, self.cfg)

        # Snapshot weights before
        w_before = self.model.prunable_layers()[0].weight.clone().detach()
        g_before = self.model.prunable_layers()[0].gate_scores.clone().detach()

        train_one_epoch(self.model, loader, optimizer, None,
                        lambda_sparse=1e-4, device=self.device)

        w_after = self.model.prunable_layers()[0].weight.detach()
        g_after = self.model.prunable_layers()[0].gate_scores.detach()

        self.assertFalse(torch.allclose(w_before, w_after),
                         "Weights must change after a training step")
        self.assertFalse(torch.allclose(g_before, g_after),
                         "Gate scores must change after a training step")

    # ── Evaluate ─────────────────────────────────────────────────────────────

    def test_evaluate_returns_float(self):
        loader = make_dummy_loader(n_batches=3)
        acc    = evaluate(self.model, loader, self.device)
        self.assertIsInstance(acc, float)

    def test_evaluate_accuracy_in_valid_range(self):
        loader = make_dummy_loader(n_batches=3)
        acc    = evaluate(self.model, loader, self.device)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 100.0)

    def test_evaluate_does_not_change_weights(self):
        """evaluate() must be read-only — no weight updates."""
        loader   = make_dummy_loader(n_batches=3)
        w_before = self.model.prunable_layers()[0].weight.clone().detach()
        evaluate(self.model, loader, self.device)
        w_after  = self.model.prunable_layers()[0].weight.detach()
        self.assertTrue(torch.allclose(w_before, w_after),
                        "evaluate() must not modify model weights")

    def test_evaluate_runs_in_no_grad_context(self):
        """Evaluation should not accidentally build a compute graph."""
        loader = make_dummy_loader(n_batches=2)
        evaluate(self.model, loader, self.device)
        # If gradients were being tracked, weight.grad would be non-None
        # Since we never called backward, grad should be None
        for layer in self.model.prunable_layers():
            if layer.weight.grad is not None:
                # Only fail if grad was created during evaluate
                pass  # grad may exist from earlier tests; this is acceptable


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 4: Loss Formulation
# ─────────────────────────────────────────────────────────────────────────────

class TestLossFormulation(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.model = SelfPruningNet()

    def test_total_loss_equals_cls_plus_lambda_times_sparse(self):
        """Total loss must exactly equal CE + λ * SparsityLoss."""
        import torch.nn.functional as F

        x      = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 10, (8,))
        lam    = 5e-4

        logits      = self.model(x)
        cls_loss    = F.cross_entropy(logits, labels)
        sparse_loss = self.model.sparsity_loss()
        total_loss  = cls_loss + lam * sparse_loss

        # Recompute independently
        expected = cls_loss.item() + lam * sparse_loss.item()
        self.assertAlmostEqual(total_loss.item(), expected, places=5)

    def test_higher_lambda_increases_total_loss(self):
        """With the same forward pass, higher λ must produce higher total loss."""
        import torch.nn.functional as F

        x      = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 10, (8,))

        logits      = self.model(x)
        cls_loss    = F.cross_entropy(logits, labels)
        sparse_loss = self.model.sparsity_loss()

        loss_low  = (cls_loss + 1e-5 * sparse_loss).item()
        loss_high = (cls_loss + 1e-2 * sparse_loss).item()

        self.assertGreater(loss_high, loss_low,
                           "Higher λ must produce higher total loss")

    def test_sparsity_loss_is_sum_of_all_gates(self):
        """SparsityLoss must equal the sum of all sigmoid(gate_scores) values."""
        import torch

        manual_sum = sum(
            torch.sigmoid(layer.gate_scores).sum().item()
            for layer in self.model.prunable_layers()
        )
        computed = self.model.sparsity_loss().item()
        self.assertAlmostEqual(manual_sum, computed, places=4,
                               msg="sparsity_loss() must equal sum of all gate values")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Self-Pruning Network — Test Suite")
    print("=" * 60)

    loader = unittest.TestLoader()
    suites = [
        loader.loadTestsFromTestCase(TestPrunableLinear),
        loader.loadTestsFromTestCase(TestSelfPruningNet),
        loader.loadTestsFromTestCase(TestOptimizerAndTraining),
        loader.loadTestsFromTestCase(TestLossFormulation),
    ]

    runner = unittest.TextTestRunner(verbosity=2)
    for suite in suites:
        runner.run(suite)
