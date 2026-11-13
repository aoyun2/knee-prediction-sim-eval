"""Load the MoCapAct expert reference bank."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import EvalConfig
from .experts import discover_expert_snippets, ensure_full_expert_zoo
from .reference_bank import ExpertSnippetBank, build_expert_snippet_bank


def _npz_has_key(path: Path, key: str) -> bool:
    try:
        with np.load(path, allow_pickle=True) as d:
            return bool(key in getattr(d, "files", ()))
    except Exception:
        return False


def _ensure_reference_bank(cfg: EvalConfig, *, out_root: Path) -> ExpertSnippetBank:
    ensure_full_expert_zoo(experts_root=cfg.experts_root, downloads_dir=cfg.experts_downloads_dir)
    snippets = discover_expert_snippets(cfg.experts_root)
    print(f"[mocap_phys_eval] status: experts discovered (n_snippets={len(snippets)})")
    if len(snippets) < 2500:
        raise RuntimeError(
            f"Expected ~2589 expert snippets, but found only {len(snippets)} under {str(cfg.experts_root)!r}. "
            "This usually means the expert zoo download/extraction is incomplete."
        )

    bank_path = out_root / "reference_bank" / "mocapact_expert_snippets_right.npz"
    bank_ok = False
    if bank_path.exists():
        try:
            print(f"[mocap_phys_eval] status: loading reference bank ({bank_path})")
            bank = ExpertSnippetBank.load_npz(bank_path)
            bank_ok = bool(len(bank) >= int(len(snippets))) and _npz_has_key(bank_path, "thigh_anat_quat_world_wxyz")
        except Exception:
            bank_ok = False
    if not bank_ok:
        print("[mocap_phys_eval] status: building expert snippet reference bank (this may take a while)")
        bank = build_expert_snippet_bank(experts_root=cfg.experts_root, side="right")
        bank.save_npz(bank_path)
    return bank
