"""Standard-library tests for experiment configuration."""

import unittest
from pathlib import Path

from src.utils.config import load_config


class ExperimentConfigTests(unittest.TestCase):
    def test_loads_complete_grounded_scan_json(self):
        path = Path("configs/experiments/scan_grounded.json")

        config = load_config(path)

        self.assertEqual(config.data.dataset, "scan")
        self.assertEqual(config.data.data_dir, "data/scan")
        self.assertEqual(config.data.validation_fraction, 0.1)
        self.assertEqual(config.data.input_representation, "plain")
        self.assertFalse(config.data.nonce_primitives)
        self.assertEqual(config.data.structure_corruption_probability, 0.0)
        self.assertTrue(config.model.pretrained)
        self.assertFalse(config.model.cross_layer_consistency)
        self.assertGreater(config.abstraction.composition_weight, 0.0)
        self.assertEqual(config.training.seed, 42)

    def test_matched_controls_change_only_auxiliary_objectives(self):
        expected_weights = {
            "scan_bottleneck.json": (1.0, 0.0, 0.0, 0.0, 0.0),
            "scan_bottleneck_entropy.json": (1.0, 0.0, 0.1, 0.0, 0.0),
            "scan_contrastive.json": (0.0, 0.0, 0.0, 1.0, 0.0),
            "scan_no_auxiliary.json": (0.0, 0.0, 0.0, 0.0, 0.0),
        }
        base = load_config("configs/experiments/scan_grounded.json")

        for filename, expected in expected_weights.items():
            with self.subTest(filename=filename):
                control = load_config(Path("configs/experiments") / filename)
                actual = (
                    control.abstraction.concretization_weight,
                    control.abstraction.composition_weight,
                    control.abstraction.entropy_regularization,
                    control.abstraction.contrastive_weight,
                    control.abstraction.structural_contrastive_weight,
                )
                self.assertEqual(actual, expected)
                self.assertEqual(control.model, base.model)
                self.assertEqual(control.data, base.data)
                self.assertEqual(control.training, base.training)

    def test_structural_ablations_change_only_structure_mode(self):
        base = load_config("configs/experiments/scan_full_contrastive.json")

        for filename, expected_mode in (
            ("scan_shuffled_structure.json", "shuffled"),
            ("scan_random_structure.json", "random"),
        ):
            with self.subTest(filename=filename):
                ablation = load_config(Path("configs/experiments") / filename)
                self.assertEqual(
                    ablation.data.composition_structure_mode, expected_mode
                )
                ablation.data.composition_structure_mode = "grounded"
                self.assertEqual(ablation.data, base.data)
                self.assertEqual(ablation.model, base.model)
                self.assertEqual(ablation.training, base.training)
                self.assertEqual(ablation.abstraction, base.abstraction)

    def test_loss_ablations_change_only_one_objective_weight(self):
        base = load_config("configs/experiments/scan_full_contrastive.json")

        for filename, field_name in (
            ("scan_no_composition_loss.json", "composition_weight"),
            ("scan_no_entropy.json", "entropy_regularization"),
            ("scan_no_reconstruction.json", "concretization_weight"),
        ):
            with self.subTest(filename=filename):
                ablation = load_config(Path("configs/experiments") / filename)
                self.assertEqual(getattr(ablation.abstraction, field_name), 0.0)
                setattr(
                    ablation.abstraction,
                    field_name,
                    getattr(base.abstraction, field_name),
                )
                self.assertEqual(ablation.abstraction, base.abstraction)
                self.assertEqual(ablation.model, base.model)
                self.assertEqual(ablation.data, base.data)
                self.assertEqual(ablation.training, base.training)

    def test_layer_ablations_change_only_constrained_layers(self):
        base = load_config("configs/experiments/scan_full_contrastive.json")

        for filename, expected_layers in (
            ("scan_layer_2_only.json", [2]),
            ("scan_layer_4_only.json", [4]),
        ):
            with self.subTest(filename=filename):
                ablation = load_config(Path("configs/experiments") / filename)
                self.assertEqual(ablation.model.constrained_layers, expected_layers)
                ablation.model.constrained_layers = base.model.constrained_layers
                self.assertEqual(ablation.model, base.model)
                self.assertEqual(ablation.data, base.data)
                self.assertEqual(ablation.training, base.training)
                self.assertEqual(ablation.abstraction, base.abstraction)

    def test_composition_rule_ablations_change_only_declared_model_flag(self):
        base = load_config("configs/experiments/scan_full_contrastive.json")

        for filename, field_name in (
            ("scan_frozen_random_composition_rules.json", "composition_rules_trainable"),
            ("scan_operator_agnostic.json", "operator_specific_composition"),
        ):
            with self.subTest(filename=filename):
                ablation = load_config(Path("configs/experiments") / filename)
                self.assertFalse(getattr(ablation.model, field_name))
                setattr(ablation.model, field_name, getattr(base.model, field_name))
                self.assertEqual(ablation.model, base.model)
                self.assertEqual(ablation.data, base.data)
                self.assertEqual(ablation.training, base.training)
                self.assertEqual(ablation.abstraction, base.abstraction)

    def test_structural_objective_configs_change_only_objective_selection(self):
        base = load_config("configs/experiments/scan_grounded.json")

        for filename, objective in (
            ("scan_structure_mse.json", "mse"),
            ("scan_structure_cosine.json", "cosine"),
        ):
            with self.subTest(filename=filename):
                ablation = load_config(Path("configs/experiments") / filename)
                self.assertEqual(ablation.abstraction.composition_objective, objective)
                ablation.abstraction.composition_objective = "domain"
                self.assertEqual(ablation.abstraction, base.abstraction)
                self.assertEqual(ablation.model, base.model)
                self.assertEqual(ablation.data, base.data)
                self.assertEqual(ablation.training, base.training)

        contrastive = load_config(
            "configs/experiments/scan_structure_contrastive.json"
        )
        self.assertEqual(contrastive.abstraction.composition_weight, 0.0)
        self.assertEqual(
            contrastive.abstraction.structural_contrastive_weight,
            base.abstraction.composition_weight,
        )

    def test_full_minus_structural_contrastive_is_a_one_factor_ablation(self):
        base = load_config("configs/experiments/scan_full_contrastive.json")
        ablation = load_config(
            "configs/experiments/scan_no_structural_contrastive.json"
        )
        self.assertEqual(ablation.abstraction.structural_contrastive_weight, 0.0)
        ablation.abstraction.structural_contrastive_weight = (
            base.abstraction.structural_contrastive_weight
        )
        self.assertEqual(ablation.abstraction, base.abstraction)
        self.assertEqual(ablation.model, base.model)
        self.assertEqual(ablation.data, base.data)
        self.assertEqual(ablation.training, base.training)

    def test_primary_scan_methods_encode_distinct_scientific_controls(self):
        config_dir = Path("configs/experiments")
        reference = load_config(config_dir / "scan_reference_t5.json")
        random_init = load_config(config_dir / "scan_random_init_t5.json")
        linearized = load_config(config_dir / "scan_tree_linearized.json")
        simple = load_config(config_dir / "scan_simple_consistency.json")
        proposed = load_config(config_dir / "scan_full_contrastive.json")

        self.assertEqual(reference.model.architecture, "reference_t5")
        self.assertTrue(reference.model.pretrained)
        self.assertEqual(reference.abstraction.abstraction_loss_weight, 0.0)
        self.assertFalse(random_init.model.pretrained)
        self.assertEqual(linearized.data.input_representation, "tree_linearized")
        self.assertEqual(simple.abstraction.composition_weight, 1.0)
        self.assertEqual(simple.abstraction.structural_contrastive_weight, 0.0)
        self.assertGreater(
            proposed.abstraction.structural_contrastive_weight, 0.0
        )
        self.assertEqual(proposed.model.domain_type, "type")
        self.assertTrue(proposed.model.operator_specific_composition)
        self.assertTrue(proposed.model.require_grounded_composition)

    def test_numerical_smoke_is_bounded_and_exercises_structure(self):
        smoke = load_config("configs/experiments/scan_numerical_smoke.json")

        self.assertEqual(smoke.training.max_steps, 200)
        self.assertTrue(smoke.training.fp16)
        self.assertEqual(smoke.training.fp16_initial_scale, 1024.0)
        self.assertLess(smoke.abstraction.warmup_steps, smoke.training.max_steps)
        self.assertGreater(smoke.abstraction.structural_contrastive_weight, 0.0)

    def test_training_routes_slog_to_official_data_module(self):
        from scripts.train import DATASET_MODULES, get_data_module
        from src.data.slog_dataset import SLOGDataModule

        config = load_config("configs/experiments/scan_grounded.json")
        config.data.data_dir = "data/slog"
        module = get_data_module("slog", object(), config)

        self.assertIs(DATASET_MODULES["slog"], SLOGDataModule)
        self.assertIsInstance(module, SLOGDataModule)
        self.assertEqual(module.data_dir, "data/slog")


if __name__ == "__main__":
    unittest.main()
