"""Synthetic checkpoint naming, legacy recovery, and retention contracts."""
from pathlib import Path

import pytest
import torch

import train
from utils.checkpoint import dit_resume_candidates, has_dit_checkpoint
from utils.eval_outputs import eval_file_base, eval_ref_suffix


def test_checkpoint_save_and_resume_use_dit_prefix(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward(); optimizer.step(); optimizer.zero_grad()
    ema = train.EMA(model)
    train.save_ckpt(model, optimizer, 1, tmp_path, ema=ema)
    train.save_best_ckpt(model, tmp_path, 0.5, ema=ema)
    ck = tmp_path / 'ckpts'
    assert {p.name for p in ck.iterdir()} == {
        'dit_ckpt_step_1.pt', 'dit_ckpt_step_1_ema.pt',
        'dit_ckpt_step_latest.pt', 'dit_ckpt_step_latest_ema.pt',
        'dit_ckpt_step_best.pt', 'dit_ckpt_step_best_ema.pt',
    }
    assert has_dit_checkpoint(ck)
    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters())
    restored_ema = train.EMA(restored)
    assert train.maybe_resume(restored, restored_optimizer, tmp_path, {}, restored_ema) == 1
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b)


def test_corrupt_new_latest_falls_back_before_stale_legacy_latest(tmp_path):
    ck = tmp_path / 'ckpts'; ck.mkdir()
    (ck / 'dit_ckpt_step_latest.pt').write_bytes(b'INTENTIONALLY INVALID SYNTHETIC CHECKPOINT')
    torch.save({'step': 20}, ck / 'dit_ckpt_step_20.pt')
    torch.save({'step': 10}, ck / 'ckpt_step_latest.pt')
    torch.save({'step': 10}, ck / 'ckpt_step_10.pt')
    torch.save({'step': 999}, ck / 'vae_ckpt_step_999.pt')
    paths = dit_resume_candidates(ck)
    assert [p.name for p in paths] == ['dit_ckpt_step_latest.pt', 'dit_ckpt_step_20.pt',
                                      'ckpt_step_latest.pt', 'ckpt_step_10.pt']
    path, payload = train._load_first_valid(paths)
    assert path.name == 'dit_ckpt_step_20.pt' and payload['step'] == 20
    assert dit_resume_candidates(ck, 10)[0].name == 'ckpt_step_10.pt'
    assert dit_resume_candidates(ck, 20)[0].name == 'dit_ckpt_step_20.pt'


def test_prune_counts_steps_and_never_removes_vae_or_latest(tmp_path):
    tmp_path = tmp_path / 'ckpts'
    tmp_path.mkdir()
    names = ['ckpt_step_10.pt', 'ckpt_step_10_ema.pt',
             'ckpt_step_20.pt', 'ckpt_step_20_ema.pt',
             'dit_ckpt_step_20.pt', 'dit_ckpt_step_20_ema.pt',
             'dit_ckpt_step_30.pt', 'dit_ckpt_step_30_ema.pt',
             'dit_ckpt_step_latest.pt', 'dit_ckpt_step_best.pt',
             'vae_ckpt_step_1.pt', 'vae_ckpt_step_1_ema.pt',
             'dit_ckpt_step_40.pt.tmp']
    for name in names:
        (tmp_path / name).touch()
    train._prune_old_ckpts(tmp_path, 2)
    assert {p.name for p in tmp_path.iterdir()} == set(names) - {'ckpt_step_10.pt', 'ckpt_step_10_ema.pt'}


def test_vae_and_ema_exports_do_not_trigger_auto_resume(tmp_path):
    for name in ('vae_ckpt_step_latest.pt', 'vae_ckpt_step_8000.pt',
                 'dit_ckpt_step_20_ema.pt', 'dit_ckpt_step_best.pt'):
        (tmp_path / name).touch()
    assert not has_dit_checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError):
        dit_resume_candidates(tmp_path)


@pytest.mark.parametrize('task,mode,directory', [
    ('sr', 'sr', 'sr'), ('sr', 'sr_native_thick', 'sr/native_thick'),
    ('deblur', 'deblur_native_thick', 'deblur/native_thick'),
    ('dealias', 'dealias_native_thick', 'dealias/native_thick'),
    ('inpaint', 'inpaint', 'inpaint/healthy_region'),
    ('inpaint', 'inpaint_tumor', 'inpaint/tumor_region'),
    ('inpaint', 'inpaint_lesion', 'inpaint/lesion_rebuild'),
])
def test_eval_modes_have_distinct_searchable_directories(task, mode, directory):
    base = eval_file_base(task, mode, 'case000001', 1, 'val2')
    assert Path(base).parent.as_posix() == directory
    assert '__' not in base and '_val000002_' in base
    assert '__' not in eval_ref_suffix(1, 2, 'masked_image', 1, decoded=True)
