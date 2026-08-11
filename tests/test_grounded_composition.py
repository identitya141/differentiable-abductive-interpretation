"""Focused tests for grounded abstract composition constraints."""

import unittest
from unittest.mock import patch

import torch
from transformers import T5Config, T5ForConditionalGeneration

from src.data.scan_composition import SCANCompositionSpec
from src.losses.abstraction_loss import AbstractionLoss, OverConstraintDetector
from src.models.abstract_domains import (
    IntervalDomain,
    MonotonicityDomain,
    TypeElement,
    TypeDomain,
    TypeMonotonicityDomain,
)
from src.models.dai_transformer import DAIConfig, DAIEncoderWrapper, DAITransformer
from src.models.abstraction_layer import AbstractionScheduler


class GroundedCompositionLossTests(unittest.TestCase):
    def test_contrastive_type_features_are_canonical_fp32_probabilities(self):
        logits = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float16)
        shifted = logits + 100.0

        first = AbstractionLoss._contrastive_features(
            TypeElement(logits), torch.zeros(1, 1, 3)
        )
        second = AbstractionLoss._contrastive_features(
            TypeElement(shifted), torch.zeros(1, 1, 3)
        )

        self.assertEqual(first.dtype, torch.float32)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first.sum(dim=-1), torch.ones(1))

    def test_type_concretization_ignores_padding_and_entropy_is_not_double_counted(self):
        torch.manual_seed(3)
        domain = TypeDomain(hidden_dim=8, num_types=4, type_embed_dim=4)
        valid = torch.randn(1, 2, 8)
        first = torch.cat([valid, torch.zeros(1, 2, 8)], dim=1)
        second = torch.cat([valid, torch.full((1, 2, 8), 100.0)], dim=1)
        mask = torch.tensor([[1, 1, 0, 0]])
        first_loss = domain.concretize_loss(
            first, domain.abstract(first), attention_mask=mask
        )
        second_loss = domain.concretize_loss(
            second, domain.abstract(second), attention_mask=mask
        )
        torch.testing.assert_close(first_loss, second_loss)

    def test_monotonicity_and_coupling_losses_are_active_and_shape_aligned(self):
        torch.manual_seed(5)
        hidden = torch.randn(2, 5, 8, requires_grad=True)
        domain = TypeMonotonicityDomain(
            hidden_dim=8, num_types=4, type_embed_dim=4, monotonicity_dim=4
        )
        abstraction = domain.abstract(hidden)
        self.assertEqual(abstraction.coupling_logits.shape, torch.Size([2, 4, 3]))
        self.assertEqual(
            abstraction.monotonicity_component.monotonicity_logits.shape,
            torch.Size([2, 4, 3]),
        )
        loss = domain.concretize_loss(
            hidden, abstraction, attention_mask=torch.ones(2, 5)
        )
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(domain.type_mono_coupling.grad)

    def test_interval_concretization_is_not_zero_by_construction(self):
        torch.manual_seed(9)
        domain = IntervalDomain(hidden_dim=8, interval_dim=4)
        hidden = torch.randn(2, 3, 8)
        loss = domain.concretize_loss(hidden, domain.abstract(hidden))
        self.assertGreater(loss.item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_product_domain_autocast_backward_has_finite_gradients(self):
        domain = TypeMonotonicityDomain(
            hidden_dim=32,
            num_types=8,
            type_embed_dim=16,
            monotonicity_dim=16,
        ).cuda()
        loss_fn = AbstractionLoss(
            abstract_domain=domain,
            concretization_weight=1.0,
            composition_weight=0.5,
            entropy_regularization=0.1,
            structural_contrastive_weight=0.5,
        ).cuda()
        hidden_states = torch.randn(
            4, 6, 32, device="cuda", requires_grad=True
        )
        specs = [
            [
                SCANCompositionSpec((0, 1), (1, 2), (0, 2), "and"),
                SCANCompositionSpec((2, 3), (3, 4), (2, 4), "after"),
            ]
            for _ in range(4)
        ]

        with torch.autocast("cuda", dtype=torch.float16):
            output = loss_fn(
                hidden_states,
                attention_mask=torch.ones(4, 6, device="cuda"),
                composition_specs=specs,
            )
        output.total_loss.backward()

        self.assertTrue(torch.isfinite(output.total_loss))
        gradients = [
            parameter.grad
            for parameter in loss_fn.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(torch.isfinite(hidden_states.grad).all())

    def test_pairwise_composition_objectives_are_finite_and_differentiable(self):
        specs = [[SCANCompositionSpec((0, 1), (1, 2), (0, 2), "and")]]

        for objective in ("domain", "mse", "cosine"):
            with self.subTest(objective=objective):
                torch.manual_seed(7)
                domain = TypeDomain(hidden_dim=8, num_types=4, type_embed_dim=4)
                hidden_states = torch.randn(1, 3, 8, requires_grad=True)
                output = AbstractionLoss(
                    abstract_domain=domain,
                    concretization_weight=0.0,
                    composition_weight=1.0,
                    entropy_regularization=0.0,
                    composition_objective=objective,
                )(
                    hidden_states=hidden_states,
                    attention_mask=torch.ones(1, 3),
                    composition_specs=specs,
                )

                self.assertTrue(torch.isfinite(output.total_loss))
                output.total_loss.backward()
                self.assertIsNotNone(hidden_states.grad)
                self.assertGreater(hidden_states.grad.norm().item(), 0.0)

    def test_rejects_unknown_composition_objective(self):
        with self.assertRaisesRegex(ValueError, "composition_objective"):
            AbstractionLoss(
                TypeDomain(hidden_dim=8, num_types=4, type_embed_dim=4),
                composition_objective="unsupported",
            )

    def test_fixed_composition_rules_do_not_receive_gradients(self):
        domain = TypeDomain(
            hidden_dim=8,
            num_types=4,
            type_embed_dim=4,
            composition_rules_trainable=False,
        )
        loss_fn = AbstractionLoss(
            abstract_domain=domain,
            concretization_weight=0.0,
            composition_weight=1.0,
            entropy_regularization=0.0,
        )
        output = loss_fn(
            hidden_states=torch.randn(1, 3, 8, requires_grad=True),
            attention_mask=torch.ones(1, 3),
            composition_specs=[
                [SCANCompositionSpec((0, 1), (1, 2), (0, 2), "and")]
            ],
        )

        output.total_loss.backward()

        self.assertFalse(domain.composition_rules.requires_grad)
        self.assertFalse(domain.operator_composition_rules["and"].requires_grad)
        self.assertIsNone(domain.operator_composition_rules["and"].grad)

    def test_operator_agnostic_composition_uses_shared_rules(self):
        domain = TypeDomain(
            hidden_dim=8,
            num_types=4,
            type_embed_dim=4,
            operator_specific_composition=False,
        )
        loss_fn = AbstractionLoss(
            abstract_domain=domain,
            concretization_weight=0.0,
            composition_weight=1.0,
            entropy_regularization=0.0,
        )
        output = loss_fn(
            hidden_states=torch.randn(1, 3, 8, requires_grad=True),
            attention_mask=torch.ones(1, 3),
            composition_specs=[
                [SCANCompositionSpec((0, 1), (1, 2), (0, 2), "and")]
            ],
        )

        output.total_loss.backward()

        self.assertIsNotNone(domain.composition_rules.grad)
        self.assertGreater(domain.composition_rules.grad.norm().item(), 0.0)
        self.assertIsNone(domain.operator_composition_rules["and"].grad)

    def test_span_composition_produces_loss_and_operator_gradients(self):
        torch.manual_seed(7)
        domain = TypeDomain(hidden_dim=8, num_types=4, type_embed_dim=4)
        loss_fn = AbstractionLoss(
            abstract_domain=domain,
            concretization_weight=0.0,
            composition_weight=1.0,
            entropy_regularization=0.0,
        )
        hidden_states = torch.randn(1, 3, 8, requires_grad=True)
        specs = [[SCANCompositionSpec((0, 1), (1, 2), (0, 2), "and")]]

        output = loss_fn(
            hidden_states=hidden_states,
            attention_mask=torch.ones(1, 3),
            composition_specs=specs,
        )
        output.total_loss.backward()

        self.assertGreater(output.total_loss.item(), 0.0)
        self.assertEqual(output.loss_components["composition_count"].item(), 1.0)
        self.assertEqual(
            output.loss_components["composition_count_per_example"].tolist(),
            [1.0],
        )
        self.assertEqual(
            output.loss_components["composition_per_example"].shape,
            torch.Size([1]),
        )
        self.assertIsNotNone(hidden_states.grad)
        self.assertGreater(hidden_states.grad.norm().item(), 0.0)
        operator_gradient = domain.operator_composition_rules["and"].grad
        self.assertIsNotNone(operator_gradient)
        self.assertGreater(operator_gradient.norm().item(), 0.0)
        self.assertIsNone(domain.operator_composition_rules["after"].grad)

    def test_rejects_spans_past_valid_tokens(self):
        domain = TypeDomain(hidden_dim=8, num_types=4, type_embed_dim=4)
        loss_fn = AbstractionLoss(domain)
        specs = [[SCANCompositionSpec((0, 1), (1, 3), (0, 3), "and")]]

        with self.assertRaises(ValueError):
            loss_fn(
                hidden_states=torch.randn(1, 4, 8),
                attention_mask=torch.tensor([[1, 1, 0, 0]]),
                composition_specs=specs,
            )

    def test_product_domain_composition_is_differentiable(self):
        torch.manual_seed(11)
        domain = TypeMonotonicityDomain(
            hidden_dim=8,
            num_types=4,
            type_embed_dim=4,
            monotonicity_dim=4,
        )
        loss_fn = AbstractionLoss(
            abstract_domain=domain,
            concretization_weight=0.0,
            composition_weight=1.0,
            entropy_regularization=0.0,
        )
        hidden_states = torch.randn(1, 3, 8, requires_grad=True)

        output = loss_fn(
            hidden_states=hidden_states,
            attention_mask=torch.ones(1, 3),
            composition_specs=[
                [SCANCompositionSpec((0, 1), (1, 2), (0, 2), "twice")]
            ],
        )
        output.total_loss.backward()

        classifier = domain.monotonicity_domain.monotonicity_classifier[-1]
        self.assertIsNotNone(classifier.weight.grad)
        self.assertGreater(classifier.weight.grad.norm().item(), 0.0)

    def test_contrastive_control_uses_operator_positive_pairs(self):
        torch.manual_seed(13)
        domain = TypeDomain(hidden_dim=8, num_types=4, type_embed_dim=4)
        loss_fn = AbstractionLoss(
            abstract_domain=domain,
            concretization_weight=0.0,
            composition_weight=0.0,
            entropy_regularization=0.0,
            contrastive_weight=1.0,
            contrastive_temperature=0.2,
        )
        hidden_states = torch.randn(4, 3, 8, requires_grad=True)
        operators = ("and", "and", "after", "after")
        specs = [
            [SCANCompositionSpec((0, 1), (1, 2), (0, 2), operator)]
            for operator in operators
        ]

        output = loss_fn(
            hidden_states=hidden_states,
            attention_mask=torch.ones(4, 3),
            composition_specs=specs,
        )
        output.total_loss.backward()

        self.assertGreater(output.loss_components["contrastive"].item(), 0.0)
        self.assertEqual(output.loss_components["contrastive_count"].item(), 4.0)
        abstraction_gradient = domain.abstraction_net[0].weight.grad
        self.assertIsNotNone(abstraction_gradient)
        self.assertGreater(abstraction_gradient.norm().item(), 0.0)

    def test_structural_contrastive_matches_composed_children_to_parent(self):
        torch.manual_seed(23)
        domain = TypeDomain(hidden_dim=8, num_types=4, type_embed_dim=4)
        loss_fn = AbstractionLoss(
            abstract_domain=domain,
            concretization_weight=0.0,
            composition_weight=0.0,
            entropy_regularization=0.0,
            structural_contrastive_weight=1.0,
            contrastive_temperature=0.2,
        )
        hidden_states = torch.randn(3, 4, 8, requires_grad=True)
        specs = [
            [SCANCompositionSpec((0, 1), (1, 2), (0, 2), operator)]
            for operator in ("and", "after", "twice")
        ]

        output = loss_fn(
            hidden_states=hidden_states,
            attention_mask=torch.ones(3, 4),
            composition_specs=specs,
        )
        output.total_loss.backward()

        self.assertGreater(
            output.loss_components["structural_contrastive"].item(), 0.0
        )
        self.assertEqual(
            output.loss_components["structural_contrastive_count"].item(), 3.0
        )
        self.assertIsNotNone(hidden_states.grad)
        self.assertGreater(hidden_states.grad.norm().item(), 0.0)
        self.assertIsNotNone(domain.operator_composition_rules["and"].grad)


class GroundedCompositionActivationTests(unittest.TestCase):
    def test_eval_diagnostics_refresh_without_adding_abstraction_loss(self):
        torch.manual_seed(29)
        t5_config = T5Config(
            vocab_size=32, d_model=8, d_ff=16, num_layers=1,
            num_heads=2, d_kv=4, dropout_rate=0.0,
            pad_token_id=0, eos_token_id=1, decoder_start_token_id=0,
        )
        with patch(
            "src.models.dai_transformer.T5ForConditionalGeneration.from_pretrained",
            return_value=T5ForConditionalGeneration(t5_config),
        ):
            model = DAITransformer(DAIConfig(
                constrained_layers=[0], domain_type="type",
                num_types=4, type_embed_dim=4,
                composition_weight=1.0,
            ))
        model.eval()
        output = model(
            input_ids=torch.tensor([[2, 3, 4]]),
            attention_mask=torch.ones(1, 3),
            labels=torch.tensor([[5, 6, 1]]),
            composition_specs=[
                [SCANCompositionSpec((0, 1), (1, 2), (0, 2), "and")]
            ],
            compute_abstraction_loss=False,
            compute_abstraction_diagnostics=True,
        )
        self.assertIsNone(output.abstraction_loss)
        self.assertIsNotNone(output.abstraction_diagnostics)
        self.assertEqual(
            output.abstraction_diagnostics["layer_0"][
                "loss_composition_count"
            ].item(),
            1.0,
        )

    def test_over_constraint_history_remains_full_after_wraparound(self):
        detector = OverConstraintDetector(
            task_loss_window=4, task_loss_increase_threshold=0.1
        )
        for value in (1.0, 1.0, 2.0, 2.0, 3.0):
            detections = detector.update(torch.tensor(value), torch.tensor(0.0))

        self.assertEqual(detector.num_history_entries.item(), 4)
        self.assertEqual(detector.history_pointer.item(), 1)
        self.assertTrue(detections["task_loss_increasing"])

    def test_over_constraint_detector_accepts_measured_gradient_norms(self):
        detector = OverConstraintDetector(gradient_ratio_threshold=2.0)
        detections = detector.update(
            task_loss=torch.tensor(1.0),
            abstraction_loss=torch.tensor(1.0),
            task_gradients=torch.tensor(1.0),
            abstraction_gradients=torch.tensor(3.0),
        )
        self.assertTrue(detections["gradient_ratio_high"])

    def test_abstraction_scheduler_backoff_state_round_trips(self):
        scheduler = AbstractionScheduler(
            use_step_schedule=True, warmup_steps=0,
            backoff_trigger_count=1, backoff_steps=10,
        )
        self.assertTrue(scheduler.notify_warning(7))
        restored = AbstractionScheduler(use_step_schedule=True, warmup_steps=0)
        restored.load_state_dict(scheduler.state_dict())
        self.assertEqual(restored.state_dict(), scheduler.state_dict())
        self.assertEqual(restored.get_weight(0, global_step=8), 0.0)

    def test_zero_weight_warmup_is_a_true_task_only_pass(self):
        t5_config = T5Config(
            vocab_size=32, d_model=8, d_ff=16, num_layers=1,
            num_decoder_layers=1, num_heads=2, d_kv=4,
            dropout_rate=0.0, decoder_start_token_id=0,
            pad_token_id=0, eos_token_id=1,
        )
        config = DAIConfig(
            constrained_layers=[0], domain_type="type", num_types=4,
            type_embed_dim=4, abstraction_loss_weight=0.1,
            warmup_steps=10, ramp_steps=0, require_grounded_composition=True,
        )
        with patch.object(T5Config, "from_pretrained", return_value=t5_config):
            model = DAITransformer(config=config, pretrained=False).train()

        layer = model.dai_encoder.abstraction_module.abstraction_layers["0"]
        with patch.object(layer, "forward", wraps=layer.forward) as abstraction_forward:
            output = model(
                input_ids=torch.tensor([[2, 3, 1]]),
                attention_mask=torch.ones(1, 3),
                labels=torch.tensor([[4, 5, 1]]),
                composition_specs=[[SCANCompositionSpec((0, 1), (1, 2), (0, 2), "and")]],
            )

        abstraction_forward.assert_not_called()
        self.assertEqual(output.abstraction_loss.item(), 0.0)
        self.assertIsNone(output.raw_abstraction_loss)

    def test_tiny_dai_optimizer_step_uses_grounded_composition(self):
        torch.manual_seed(19)
        t5_config = T5Config(
            vocab_size=32,
            d_model=8,
            d_ff=16,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            d_kv=4,
            dropout_rate=0.0,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
        )
        config = DAIConfig(
            constrained_layers=[0],
            domain_type="type",
            num_types=4,
            type_embed_dim=4,
            concretization_weight=0.0,
            composition_weight=1.0,
            entropy_regularization=0.0,
            abstraction_loss_weight=0.1,
            warmup_steps=0,
            ramp_steps=0,
            require_grounded_composition=True,
        )
        with patch.object(T5Config, "from_pretrained", return_value=t5_config):
            model = DAITransformer(config=config, pretrained=False)

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        domain = model.dai_encoder.abstraction_module.abstraction_layers["0"].abstract_domain
        operator_rule = domain.operator_composition_rules["and"]
        rule_before = operator_rule.detach().clone()

        output = model(
            input_ids=torch.tensor([[2, 3, 4]]),
            attention_mask=torch.ones(1, 3),
            labels=torch.tensor([[5, 6, 1]]),
            composition_specs=[
                [SCANCompositionSpec((0, 1), (1, 2), (0, 2), "and")]
            ],
        )
        optimizer.zero_grad()
        output.loss.backward()

        diagnostics = output.abstraction_diagnostics["layer_0"]
        self.assertGreater(output.task_loss.item(), 0.0)
        self.assertGreater(output.abstraction_loss.item(), 0.0)
        self.assertEqual(diagnostics["loss_composition_count"].item(), 1.0)
        self.assertIsNotNone(operator_rule.grad)
        self.assertGreater(operator_rule.grad.norm().item(), 0.0)

        optimizer.step()
        self.assertFalse(torch.equal(rule_before, operator_rule.detach()))

    def test_required_grounded_composition_rejects_missing_metadata(self):
        t5_config = T5Config(
            vocab_size=32,
            d_model=8,
            d_ff=16,
            num_layers=1,
            num_heads=2,
            d_kv=4,
        )
        encoder = T5ForConditionalGeneration(t5_config).encoder
        config = DAIConfig(
            constrained_layers=[0],
            domain_type="type",
            num_types=4,
            type_embed_dim=4,
            require_grounded_composition=True,
        )
        wrapper = DAIEncoderWrapper(encoder, config)
        wrapper.train()

        with self.assertRaisesRegex(ValueError, "no composition metadata"):
            wrapper(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.ones(1, 3),
                composition_specs=None,
            )

        output = wrapper(
            input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
            attention_mask=torch.ones(2, 3),
            composition_specs=[[], []],
        )
        self.assertEqual(output.last_hidden_state.shape[:2], (2, 3))

    def test_t5_encoder_consumes_grounded_specs(self):
        torch.manual_seed(17)
        t5_config = T5Config(
            vocab_size=32,
            d_model=8,
            d_ff=16,
            num_layers=1,
            num_heads=2,
            d_kv=4,
            dropout_rate=0.0,
        )
        encoder = T5ForConditionalGeneration(t5_config).encoder
        config = DAIConfig(
            constrained_layers=[0],
            domain_type="type",
            num_types=4,
            type_embed_dim=4,
            concretization_weight=0.0,
            composition_weight=1.0,
            entropy_regularization=0.0,
            require_grounded_composition=True,
        )
        wrapper = DAIEncoderWrapper(encoder, config)
        wrapper.train()

        wrapper(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3),
            composition_specs=[
                [SCANCompositionSpec((0, 1), (1, 2), (0, 2), "and")]
            ],
        )
        loss = wrapper.get_total_abstraction_loss()
        loss.backward()

        diagnostics = wrapper.abstraction_module.get_all_diagnostics()["layer_0"]
        self.assertGreater(loss.item(), 0.0)
        self.assertEqual(diagnostics["loss_composition_count"].item(), 1.0)
        domain = wrapper.abstraction_module.abstraction_layers["0"].abstract_domain
        gradient = domain.operator_composition_rules["and"].grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.norm().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
