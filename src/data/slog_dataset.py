"""Official SLOG benchmark loader with category and recursion-depth metadata."""

import csv
import os
import re
from typing import Dict, List, Optional

from .base_dataset import BaseCompositionalDataset, CompositionalExample
from .cogs_composition import (
    align_cogs_composition_specs_to_tokens,
    extract_cogs_composition_specs,
)
from .composition_controls import transform_aligned_specs


SLOG_CATEGORIES = (
    "PP_3",
    "PP_5-12",
    "CP_3",
    "CP_5-12",
    "center_embed_3",
    "center_embed_5-12",
    "PP_modif_subj",
    "PP_modif_iobj",
    "RC_modif_subj",
    "RC_modif_iobj",
    "RC_iobj_extracted",
    "Q_subj_active",
    "Q_subj_passive",
    "Q_dobj_ditransV",
    "Q_iobj_ditransV",
    "Q_modified_NPs",
    "Q_long_mv",
)


def infer_slog_depth(category: str, logical_form: str) -> Optional[int]:
    """Return the benchmark recursion depth for SLOG's six depth cases."""
    if category in {"PP_3", "PP_5-12"}:
        return len(re.findall(r"\.\s*nmod\s*\.\s*\w+\s*\(", logical_form))
    if category in {"CP_3", "CP_5-12"}:
        return len(re.findall(r"\.\s*ccomp\s*\(", logical_form))
    if category in {"center_embed_3", "center_embed_5-12"}:
        return len(re.findall(r"\.\s*nmod\s*\(", logical_form))
    return None


class SLOGDataset(BaseCompositionalDataset):
    """SLOG COGS-LF data with the protected generalization split."""

    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_source_length: int = 256,
        max_target_length: int = 512,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        composition_structure_mode: str = "grounded",
        seed: int = 42,
    ):
        self.composition_structure_mode = composition_structure_mode
        self.seed = seed
        super().__init__(
            tokenizer=tokenizer,
            split=split,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            data_dir=data_dir,
            cache_dir=cache_dir,
        )

    @property
    def dataset_name(self) -> str:
        return "SLOG"

    @property
    def task_description(self) -> str:
        return "Parse structurally novel English sentences to COGS logical forms"

    def _split_path(self) -> str:
        if self.data_dir is None:
            raise ValueError("data_dir required for SLOG")
        candidates = {
            "train": ("cogs_LF/train.tsv", "train.tsv"),
            "dev": ("cogs_LF/dev.tsv", "dev.tsv"),
            "iid_test": ("cogs_LF/test.tsv", "test.tsv"),
            "test": (
                "generalization_sets/gen_cogsLF.tsv",
                "gen_cogsLF.tsv",
                "cogs_LF/gen.tsv",
                "gen.tsv",
            ),
            "gen": (
                "generalization_sets/gen_cogsLF.tsv",
                "gen_cogsLF.tsv",
                "cogs_LF/gen.tsv",
                "gen.tsv",
            ),
        }
        if self.split not in candidates:
            raise ValueError(f"Unknown SLOG split: {self.split}")
        for relative_path in candidates[self.split]:
            path = os.path.join(self.data_dir, relative_path)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            f"No official SLOG file for split {self.split!r} under {self.data_dir!r}"
        )

    def _load_data(self) -> List[CompositionalExample]:
        examples = []
        with open(self._split_path(), "r", encoding="utf-8", newline="") as handle:
            for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
                if len(row) != 3:
                    raise ValueError(
                        f"SLOG line {line_number} must have three TSV columns"
                    )
                sentence, logical_form, category = row
                if not sentence or not logical_form or not category:
                    raise ValueError(f"SLOG line {line_number} contains an empty field")
                is_generalization = self.split in {"test", "gen"}
                if is_generalization and category not in SLOG_CATEGORIES:
                    raise ValueError(
                        f"Unknown SLOG category {category!r} on line {line_number}"
                    )
                depth = infer_slog_depth(category, logical_form)
                examples.append(
                    CompositionalExample(
                        input_text=sentence,
                        target_text=logical_form,
                        split=self.split,
                        is_ood=is_generalization,
                        generalization_category=category,
                        compositional_structure=(
                            f"depth:{depth}" if depth is not None else "depth:n/a"
                        ),
                        composition_specs=list(
                            extract_cogs_composition_specs(sentence, logical_form)
                        ),
                    )
                )
        return examples

    def _tokenize_all(self) -> List[Dict]:
        tokenized = super()._tokenize_all()
        for example, encoded in zip(self.examples, tokenized):
            aligned = align_cogs_composition_specs_to_tokens(
                    example.input_text,
                    example.composition_specs or [],
                    self.tokenizer,
                )
            encoded["composition_specs"] = list(
                transform_aligned_specs(
                    aligned,
                    token_count=int(encoded["attention_mask"].sum().item()),
                    mode=self.composition_structure_mode,
                    seed=f"{self.seed}:{example.input_text}",
                )
            )
        return tokenized

    def _get_compositional_structure(self, example: CompositionalExample) -> Dict:
        value = example.compositional_structure or "depth:n/a"
        depth_text = value.split(":", 1)[1]
        return {
            "category": example.generalization_category,
            "depth": None if depth_text == "n/a" else int(depth_text),
        }

    def get_type_vocabulary(self) -> Dict[str, int]:
        operators = {"agent", "theme", "recipient", "ccomp", "xcomp", "nmod"}
        return {operator: index for index, operator in enumerate(sorted(operators))}


class SLOGDataModule:
    """Data module keeping SLOG IID and structural-generalization tests separate."""

    def __init__(
        self,
        tokenizer,
        batch_size: int = 32,
        max_source_length: int = 256,
        max_target_length: int = 512,
        num_workers: int = 4,
        eval_batch_size: Optional[int] = None,
        eval_num_workers: int = 0,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        composition_structure_mode: str = "grounded",
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_workers = num_workers
        self.eval_batch_size = eval_batch_size or batch_size
        self.eval_num_workers = eval_num_workers
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.composition_structure_mode = composition_structure_mode
        self.seed = seed

    def _dataset(self, split: str) -> SLOGDataset:
        return SLOGDataset(
            self.tokenizer,
            split=split,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            composition_structure_mode=self.composition_structure_mode,
            seed=self.seed,
        )

    def setup(self):
        self.train_dataset = self._dataset("train")
        self.dev_dataset = self._dataset("dev")
        self.iid_test_dataset = self._dataset("iid_test")
        self.test_dataset = self._dataset("test")

    def train_dataloader(self):
        return self.train_dataset.get_dataloader(
            self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return self.dev_dataset.get_dataloader(
            self.eval_batch_size, shuffle=False, num_workers=self.eval_num_workers
        )

    def validation_dataloader(self):
        return self.val_dataloader()

    def iid_test_dataloader(self):
        return self.iid_test_dataset.get_dataloader(
            self.eval_batch_size, shuffle=False, num_workers=self.eval_num_workers
        )

    def test_dataloader(self):
        return self.test_dataset.get_dataloader(
            self.eval_batch_size, shuffle=False, num_workers=self.eval_num_workers
        )
