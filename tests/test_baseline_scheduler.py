from pathlib import Path

import yaml
from transformers.trainer_utils import SchedulerType

from src.models.baselines import BASELINE_REGISTRY
from src.utils.schedulers import hf_scheduler_name


def test_every_baseline_config_has_a_valid_hf_scheduler_name():
    valid = {value.value for value in SchedulerType}
    for spec in BASELINE_REGISTRY.values():
        config = yaml.safe_load((Path("configs/baselines") / spec.config_name).read_text())
        scheduler = config.get("training", {}).get("lr_scheduler", "cosine")
        assert hf_scheduler_name(scheduler) in valid, spec.key


def test_project_cosine_alias_maps_to_hugging_face_cosine():
    assert hf_scheduler_name("cosine_with_warmup") == "cosine"
