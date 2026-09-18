"""CPU checks using fabricated arrays only; no downloads or pretrained weights."""

import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import MultiTaskDataset, multitask_collate
from models.mmdit import CrossRefFusion, MMDiT3D, MMDiTCfg
from utils.config_loader import load_train_config
from utils.h5_io import read_h5_brain_mask
from utils.prompts import acq_stem, get_seg_prompt_variants
from utils.tasks import CANONICAL_TASKS, apply_curriculum_preset, validate_task_weights

SHAPE = (24, 24, 24)
GRID = (3, 3, 3)


class SyntheticTextStore:
    def get(self, text):
        assert text.strip()
        return torch.ones(2, 8)


def write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.random.default_rng(3).uniform(0, 1, SHAPE).astype(np.float32)
    with h5py.File(path, "w") as f:
        f["image/iso1mm"] = x
        f["brain_mask/iso1mm"] = np.ones(SHAPE, np.uint8)
        f["meta/native/spacing"] = [1.0, 1.0, 1.0]
        f["meta/iso1mm/spacing"] = [1.0, 1.0, 1.0]
        g = f.require_group("latent/iso1mm/24x24x24")
        g["mean"] = np.ones((2, *GRID), np.float32)
        g["logvar"] = np.zeros((2, *GRID), np.float32)
        g.attrs.update(
            cache_version=1,
            vae_ckpt_fingerprint="synthetic",
            spatial_size=SHAPE,
            latent_shape=GRID,
        )


def make_dataset(tmp_path, task, count=4):
    label = tmp_path / "synthetic_study/labels.h5"
    label.parent.mkdir(parents=True, exist_ok=True)
    y = np.zeros(SHAPE, np.int16)
    y[8:14, 8:14, 3:7] = 1
    y[9:12, 9:12, 4:6] = 2
    with h5py.File(label, "w") as f:
        f["image/iso1mm"] = y
        g = f.require_group("latent/mask/24x24x24")
        g["mean"] = np.zeros((2, *GRID), np.float32)
        g.attrs.update(
            cache_version=1,
            vae_ckpt_fingerprint="synthetic",
            spatial_size=SHAPE,
            latent_shape=GRID,
            num_classes=4,
        )
    records, acq = [], {}
    for mod in range(count):
        bet = tmp_path / f"synthetic_study/scan_{mod}_bet.h5"
        write_image(bet)
        write_image(tmp_path / f"synthetic_study/scan_{mod}.h5")
        records.append(
            {"images": [{"path": str(bet), "modality": mod}], "label_path": str(label)}
        )
        acq[acq_stem(bet)] = ["thin", "axial", 1.0]
    manifest = tmp_path / "records.json"
    manifest.write_text(json.dumps(records))
    ds = MultiTaskDataset(
        json_path=str(manifest),
        h5_root=str(tmp_path),
        train=True,
        expected_fingerprint="synthetic",
        expected_spatial_size=SHAPE,
        expected_latent_shape=GRID,
        prompt_store=SyntheticTextStore(),
        return_label=True,
        label_h5_root=str(tmp_path),
        label_full_shape=(12, 12, 12),
        num_refs=4,
        num_modalities=5,
        latent_channels=2,
        task_weights={task: 1},
        sr_affine_enabled=False,
        sr_factors=(2,),
        sr_up_orders=(0,),
        sr_up_order_weights=(1,),
        inpaint_dilate=0,
        inpaint_margin=1,
        inpaint_pseudo_healthy_prob=0.0,
        inpaint_size_bin_edges=(),
        inpaint_size_bin_weights=(1.0,),
        inpaint_size_bin_names=("all",),
        inpaint_lesion_prob=0.0,
        region_mask_channels=1,
        acq_lookup=acq,
        seg_ref_count_weights={count: 1.0},
    )
    ds._force_task = task
    return ds


@pytest.mark.parametrize("task", CANONICAL_TASKS)
def test_training_sample_and_collation(tmp_path, task):
    random.seed(4)
    torch.manual_seed(4)
    sample = make_dataset(tmp_path, task)[0]
    assert sample["task"] == task
    assert sample["latent_mean"].shape == (2, *GRID)
    assert sample["ref_valid"].sum() <= 4
    assert torch.isfinite(sample["latent_mean"]).all()
    batch = multitask_collate([sample], fixed_text_len=8)
    assert batch["latent_mean"].shape == (1, 2, *GRID)
    assert batch["text_emb"].shape == (1, 8, 8)
    assert batch["ref_latent"].shape == (1, 4, 2, *GRID)
    if task == "seg":
        assert sample["modality_id"] == 4
        assert sample["ref_valid"].sum() == 4
    if task == "modality_only":
        assert sample["ref_valid"].sum() == 0


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_segmentation_input_subsets(tmp_path, count):
    sample = make_dataset(tmp_path, "seg", count)[0]
    assert sample["ref_valid"].sum() == count
    assert sample["modality_id"] == 4


def tiny_model():
    return MMDiT3D(
        MMDiTCfg(
            in_channels=2,
            latent_shape=GRID,
            patch_size=(1, 1, 1),
            width=48,
            text_hidden=8,
            heads=4,
            mlp_ratio=2,
            depth_double=1,
            depth_single=1,
            max_refs=4,
            num_modalities=5,
            attn_impl="sdpa",
            arch_version=8,
        )
    )


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_model_forward_backward_variable_references(count):
    torch.manual_seed(5)
    model = tiny_model()
    x = torch.randn(1, 2, *GRID)
    refs = torch.randn(1, 4, 2, *GRID)
    valid = torch.arange(4)[None, :] < count
    prediction = model(
        x,
        torch.tensor([0.5]),
        torch.randn(1, 3, 8),
        torch.ones(1, 3, dtype=torch.long),
        torch.tensor([3]),
        ref_latent=refs,
        ref_modality_id=torch.tensor([[0, 1, 2, 3]]),
        ref_valid=valid,
        ref_dt=torch.zeros(1, 4),
        ref_type_id=torch.zeros(1, 4, dtype=torch.long),
    )
    assert prediction.shape == x.shape
    loss = (prediction - torch.randn_like(x)).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )


def test_reference_fusion_masks_padding_and_preserves_permutations():
    torch.manual_seed(8)
    fusion = CrossRefFusion(48, 4)
    refs = torch.randn(1, 7, 4, 48)
    valid = torch.tensor([True, True, False, False]).expand(1, 7, 4)
    expected = fusion(refs, valid)
    altered = refs.clone()
    altered[:, :, 2:] += 1000
    torch.testing.assert_close(fusion(altered, valid), expected)
    order = [1, 0, 3, 2]
    torch.testing.assert_close(fusion(refs[:, :, order], valid[:, :, order]), expected)
    assert expected.shape == (1, 7, 48)
    assert torch.isfinite(fusion(refs, torch.zeros_like(valid))).all()


def test_missing_brain_mask_is_rejected(tmp_path):
    p = tmp_path / "synthetic.h5"
    with h5py.File(p, "w") as f:
        f["image/iso1mm"] = np.ones(SHAPE, np.float32)
    with pytest.raises((KeyError, ValueError)):
        read_h5_brain_mask(str(p), spatial_size=SHAPE)


def test_public_config_and_catalog(monkeypatch):
    for name in ("UNIGLIOMA_VAE_CHECKPOINT", "UNIGLIOMA_TEXT_MODEL"):
        monkeypatch.setenv(name, "/nonexistent/synthetic-fixture")
    cfg = load_train_config(str(ROOT / "configs/train.yaml"))
    assert cfg["use_wandb"] is False
    assert cfg["infer"]["fid_enable"] is False
    pretrain = apply_curriculum_preset(cfg, "foundation")
    assert set(pretrain["task_weights"]) == {"modality_only"}
    assert len(get_seg_prompt_variants()) == 15
    with pytest.raises(ValueError):
        validate_task_weights({"unsupported_task": 1})


def test_missing_environment_variable_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("UNIGLIOMA_TEST_MISSING", raising=False)
    p = tmp_path / "config.yaml"
    p.write_text("asset: ${UNIGLIOMA_TEST_MISSING}\n")
    with pytest.raises(ValueError, match="UNIGLIOMA_TEST_MISSING"):
        load_train_config(str(p))


@pytest.mark.parametrize("task", CANONICAL_TASKS)
def test_inference_conditioning_contract(tmp_path, monkeypatch, task):
    import infer
    from types import SimpleNamespace

    ds = make_dataset(tmp_path, task)
    paths = [r["images"][0]["path"] for r in ds.records]
    cli = [
        "--config",
        str(tmp_path / "infer.yaml"),
        "--dit-ckpt",
        "user-owned.pt",
        "--task",
        task,
    ]
    (tmp_path / "infer.yaml").write_text("inference: {}\n")
    if task == "mask_guide":
        cli += ["--label", ds.records[0]["label_h5"]]
    elif task == "missing":
        cli += ["--ref-image", *paths[:3], "--ref-modality", "0", "1", "2"]
    elif task == "seg":
        cli += ["--ref-image", *paths, "--ref-modality", "0", "1", "2", "3"]
    elif task != "modality_only":
        cli += ["--ref-image", paths[0]]
        if task == "inpaint":
            cli += [
                "--label",
                ds.records[0]["label_h5"],
                "--inpaint-content",
                "healthy",
            ]
    args = infer.parse_args(cli)
    infer.validate_cli_contract(args)
    monkeypatch.setattr(infer, "vae_ckpt_fingerprint", lambda _: "synthetic")

    def fake_encode(image, vae, cfg, device):
        image = torch.as_tensor(np.array(image, copy=True)).float()[None, None]
        return torch.nn.functional.adaptive_avg_pool3d(image, GRID).repeat(
            1, 2, 1, 1, 1
        )

    monkeypatch.setattr(infer, "_encode_image01_dhw", fake_encode)
    cfg = {
        "spatial_size": SHAPE,
        "vae_ckpt": "unused",
        "h5_mode": "iso1mm",
        "mmdit": {"latent_shape": GRID, "in_channels": 2},
    }
    model_cfg = SimpleNamespace(
        latent_shape=GRID,
        in_channels=2,
        max_refs=4,
        num_modalities=5,
        seg_num_classes=4,
    )
    cond = infer.assemble_conditioning(
        task, args, cfg=cfg, mmdit_cfg=model_cfg, vae=None, device=torch.device("cpu")
    )
    assert cond["ref_latent"].shape == (1, 4, 2, *GRID)
    assert torch.isfinite(cond["ref_latent"]).all()
    expected_count = {
        "modality_only": 0,
        "mask_guide": 1,
        "inpaint": 2,
        "whole_brain": 2,
        "missing": 3,
        "seg": 4,
    }.get(task, 1)
    assert cond["ref_valid"].sum() == expected_count
    assert cond["target_modality"] == (4 if task == "seg" else 3)


def test_inference_rejects_target_leakage(tmp_path):
    import infer

    config = tmp_path / "infer.yaml"
    config.write_text("inference: {}\n")
    args = infer.parse_args(
        [
            "--config",
            str(config),
            "--dit-ckpt",
            "user-owned.pt",
            "--task",
            "missing",
            "--target-modality",
            "3",
            "--ref-image",
            "synthetic.h5",
            "--ref-modality",
            "3",
        ]
    )
    with pytest.raises(ValueError, match="exclude"):
        infer.validate_cli_contract(args)
