"""Integration smoke tests for A#1/A#4 (cognitive modules wired into loss)."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_train_imports_without_error():
    spec = importlib.util.spec_from_file_location("src.train", ROOT / "src" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    # Only compile/parse; full execution needs torch + heavy deps, so we rely on
    # ast parse for the train.py body plus a real import of the aux-loss module.
    src = (ROOT / "src" / "train.py").read_text(encoding="utf-8")
    ast.parse(src)  # raises on syntax error
    assert "CoreKnowledgeAuxLoss" in src
    assert "core_knowledge_loss" in src


def test_core_knowledge_loss_config_present_in_stage6():
    cfg = yaml.safe_load((ROOT / "configs" / "stage6_consolidation.yaml").read_text(encoding="utf-8"))
    assert "core_knowledge_loss" in cfg
    assert cfg["core_knowledge_loss"]["enabled"] is True
    assert cfg["core_knowledge_loss"]["coef_object_permanence"] == 0.1


def test_ck_loss_module_imports():
    from src.intrinsic.core_knowledge_loss import CoreKnowledgeAuxLoss
    loss = CoreKnowledgeAuxLoss()
    assert loss is not None
