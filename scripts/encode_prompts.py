from __future__ import annotations
import argparse
import glob
import hashlib
import json
import logging
import os
import sys
import time
import h5py
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
CACHE_VERSION = 1
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("string-precompute")


def sha1(s: str) -> str:
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()


def _fsync_drop(h5file):
    try:
        h5file.flush()
        fd = h5file.id.get_vfd_handle()
    except Exception:
        return
    try:
        os.fsync(fd)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass


def validate_prompts(prompts, max_chars_hard=100000, max_chars_warn=8000, verbose=True):
    if not isinstance(prompts, list):
        raise ValueError(
            f"prompts file must be a JSON list, got {type(prompts).__name__}"
        )
    if not prompts:
        raise ValueError("prompts list is empty")
    non_str, blanks, over_hard, over_warn = ([], [], [], [])
    lens = []
    for i, x in enumerate(prompts):
        if not isinstance(x, str):
            non_str.append(i)
            continue
        n = len(x)
        lens.append(n)
        if not x.strip():
            blanks.append(i)
        if n > max_chars_hard:
            over_hard.append((i, n))
        elif n > max_chars_warn:
            over_warn.append((i, n))
    if verbose and lens:
        lens.sort()
        m = len(lens)
        log.info(
            "validate: %d prompts | char-len min/median/p99/max = %d/%d/%d/%d",
            len(prompts),
            lens[0],
            lens[m // 2],
            lens[int(m * 0.99)],
            lens[-1],
        )
    if verbose and over_warn:
        log.warning(
            "validate: %d prompt(s) > %d chars (long but allowed); first=%s",
            len(over_warn),
            max_chars_warn,
            over_warn[:3],
        )
    errs = []
    if non_str:
        errs.append(f"{len(non_str)} non-string entries (first idx {non_str[:5]})")
    if blanks:
        errs.append(
            f"{len(blanks)} blank/whitespace-only entries (first idx {blanks[:5]})"
        )
    if over_hard:
        errs.append(
            f"{len(over_hard)} entries > {max_chars_hard} chars (runaway → tokenizer stall; first {over_hard[:3]})"
        )
    if errs:
        raise ValueError("prompt validation FAILED: " + "; ".join(errs))
    if verbose:
        log.info("validate: OK (all str, no blanks, none > %d chars)", max_chars_hard)
    return {"count": len(prompts), "warn_long": len(over_warn)}


def merge_parts(out_dir, merged_path, block_rows=1000000, force=False):
    parts = sorted(glob.glob(os.path.join(out_dir, "part_*.h5")))
    if not parts:
        raise FileNotFoundError(f"no part_*.h5 in {out_dir} to merge")
    with h5py.File(parts[0], "r") as f0:
        ref_fp = str(f0.attrs.get("fingerprint", ""))
        hidden = int(f0.attrs["hidden_size"])
        base_attrs = {
            k: f0.attrs[k]
            for k in (
                "cache_version",
                "fingerprint",
                "hidden_size",
                "chat_template",
                "add_generation_prompt",
                "layer",
                "world_size",
            )
            if k in f0.attrs
        }
    if os.path.isfile(merged_path) and (not force):
        try:
            with h5py.File(merged_path, "r") as m:
                if (
                    str(m.attrs.get("fingerprint", "")) == ref_fp
                    and "feat" in m
                    and ("keys" in m)
                ):
                    log.info(
                        "merge: %s already up to date (fingerprint match) → skip [--merge-force to redo]",
                        merged_path,
                    )
                    return merged_path
        except OSError:
            pass
    total_rows, total_items = (0, 0)
    for p in parts:
        with h5py.File(p, "r") as f:
            if str(f.attrs.get("fingerprint", "")) != ref_fp:
                raise ValueError(
                    f"fingerprint mismatch in {p} — refusing to merge mixed caches"
                )
            if int(f.attrs["hidden_size"]) != hidden:
                raise ValueError(f"hidden_size mismatch in {p}")
            total_rows += int(f["feat"].shape[0])
            total_items += int(f["keys"].shape[0])
    log.info(
        "merge: %d parts → %s | %d items, %d feat rows (hidden=%d, ~%.1f GB)",
        len(parts),
        merged_path,
        total_items,
        total_rows,
        hidden,
        total_rows * hidden * 2 / 2**30,
    )
    _vlen = h5py.string_dtype(encoding="utf-8")
    tmp = merged_path + ".tmp"
    offs_all, lens_all, keys_all, feat_base = ([], [], [], 0)
    with h5py.File(tmp, "w") as g:
        for k, v in base_attrs.items():
            g.attrs[k] = v
        g.attrs["merged"] = 1
        g.attrs["n_parts"] = len(parts)
        feat = g.create_dataset(
            "feat",
            shape=(total_rows, hidden),
            dtype="float16",
            chunks=(min(4096, max(1, total_rows)), hidden),
        )
        for pi, p in enumerate(parts):
            with h5py.File(p, "r") as f:
                src = f["feat"]
                n = int(src.shape[0])
                for s in range(0, n, block_rows):
                    e = min(s + block_rows, n)
                    feat[feat_base + s : feat_base + e] = src[s:e]
                offs_all.append(np.asarray(f["offsets"], dtype=np.int64) + feat_base)
                lens_all.append(np.asarray(f["lengths"], dtype=np.int32))
                keys_all.append(np.asarray(f["keys"]))
                feat_base += n
            _fsync_drop(g)
            if pi % 10 == 0 or pi == len(parts) - 1:
                log.info(
                    "merge: %d/%d parts, %d/%d rows",
                    pi + 1,
                    len(parts),
                    feat_base,
                    total_rows,
                )
        g.create_dataset(
            "offsets",
            data=np.concatenate(offs_all) if offs_all else np.empty(0, np.int64),
        )
        g.create_dataset(
            "lengths",
            data=np.concatenate(lens_all) if lens_all else np.empty(0, np.int32),
        )
        keys_cat = np.concatenate(keys_all) if keys_all else np.empty(0, dtype=object)
        g.create_dataset("keys", data=keys_cat, dtype=_vlen)
    os.replace(tmp, merged_path)
    log.info(
        "merge: DONE → %s (%d items, %d rows)", merged_path, total_items, feat_base
    )
    return merged_path


def _dist_info():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    return (rank, world, max(0, local))


def _token_budget_batches(items, est_tokens, batch_tokens, max_batch):
    batch, cur_max = ([], 0)
    for it, n in zip(items, est_tokens):
        new_max = max(cur_max, n)
        if batch and (
            (len(batch) + 1) * new_max > batch_tokens or len(batch) >= max_batch
        ):
            yield batch
            batch, cur_max = ([], 0)
            new_max = n
        batch.append(it)
        cur_max = new_max
    if batch:
        yield batch


def _encode_batch_safe(enc, texts):
    try:
        return enc.encode_valid_list(texts)
    except torch.cuda.OutOfMemoryError:
        if len(texts) == 1:
            raise
        torch.cuda.empty_cache()
        mid = len(texts) // 2
        return _encode_batch_safe(enc, texts[:mid]) + _encode_batch_safe(
            enc, texts[mid:]
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--prompts", help="JSON list of prompt strings (required unless --merge-only)"
    )
    ap.add_argument(
        "--qwen3vl", help="Qwen3-VL model dir (required unless --merge-only)"
    )
    ap.add_argument(
        "--out",
        required=True,
        help="OUTPUT DIRECTORY for part files (part_{rank}_{chunk}.h5)",
    )
    ap.add_argument(
        "--merge-only",
        action="store_true",
        help="skip ALL encoding; just stitch existing --out/part_*.h5 into a single --merge-out file (no model / no prompts needed). Run this after a completed encode to get one big file.",
    )
    ap.add_argument(
        "--merge-out",
        default=None,
        help="single merged h5 FILE path (used with --merge-only). Point the reader's prompt_feature_h5 at this file.",
    )
    ap.add_argument(
        "--merge-force",
        action="store_true",
        help="re-merge even if --merge-out already exists with a matching fingerprint.",
    )
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument(
        "--batch-tokens",
        type=int,
        default=24000,
        help="padded-token budget per batch (drives effective batch size by length)",
    )
    ap.add_argument(
        "--max-batch", type=int, default=512, help="hard cap on batch item count"
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="prompts per output part file. Each chunk is built in RAM then written ONCE (atomic tmp+rename); a done chunk is skipped on resume. Small self-contained files avoid the ever-growing-dataset / fsync write stalls that hung the single-giant-file version, and make the job restartable without losing finished chunks.",
    )
    ap.add_argument("--chat-template", type=int, default=1)
    ap.add_argument("--add-gen", type=int, default=1)
    ap.add_argument("--layer", type=int, default=-1)
    args = ap.parse_args()
    if args.merge_only:
        if not args.merge_out:
            ap.error("--merge-only requires --merge-out")
        merge_parts(args.out, args.merge_out, force=bool(args.merge_force))
        return
    if not args.prompts or not args.qwen3vl:
        ap.error(
            "--prompts and --qwen3vl are required for encoding (or use --merge-only)"
        )
    rank, world, local = _dist_info()
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    validate_prompts(prompts, verbose=rank == 0)
    seen, uniq = (set(), [])
    for p in prompts:
        p = str(p)
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    shard = uniq[rank::world]
    n_prompts, n_uniq = (len(prompts), len(uniq))
    del prompts, seen, uniq
    if rank == 0:
        log.info(
            "%d prompts → %d unique; world=%d, ~%d per rank",
            n_prompts,
            n_uniq,
            world,
            len(shard),
        )
    shard.sort(key=len)
    est = [max(1, len(s) // 3 + 8) for s in shard]
    os.makedirs(args.out, exist_ok=True)

    def part_path(ci):
        return os.path.join(args.out, f"part_{rank}_{ci:05d}.h5")

    chunk_size = max(1, args.chunk_size)
    n_chunks = (len(shard) + chunk_size - 1) // chunk_size
    if n_chunks > 0 and all((os.path.isfile(part_path(ci)) for ci in range(n_chunks))):
        log.info(
            "rank%d: all %d chunk files present → skip encode + model load [resume]. Merge with --merge-only.",
            rank,
            n_chunks,
        )
        return
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from models.qwen3vl_text import Qwen3VLTextEncoder

    enc = Qwen3VLTextEncoder(
        model_path=args.qwen3vl,
        device=device,
        dtype=torch.bfloat16,
        chat_template=bool(args.chat_template),
        add_generation_prompt=bool(args.add_gen),
        layer=args.layer,
        max_length=args.max_length,
    )
    _vlen = h5py.string_dtype(encoding="utf-8")
    attrs = {
        "cache_version": CACHE_VERSION,
        "fingerprint": enc.fingerprint,
        "hidden_size": int(enc.hidden_size),
        "chat_template": int(args.chat_template),
        "add_generation_prompt": int(args.add_gen),
        "layer": int(args.layer),
        "rank": rank,
        "world_size": world,
    }

    def chunk_is_done(ci):
        p = part_path(ci)
        if not os.path.isfile(p):
            return False
        try:
            with h5py.File(p, "r") as g:
                return (
                    str(g.attrs.get("fingerprint", "")) == enc.fingerprint
                    and "feat" in g
                    and ("keys" in g)
                )
        except OSError:
            return False

    def write_part(ci, block, offs, lens, keys):
        tmp = part_path(ci) + ".tmp"
        with h5py.File(tmp, "w") as g:
            for k, v in attrs.items():
                g.attrs[k] = v
            g.attrs["chunk_id"] = int(ci)
            nrows = int(block.shape[0])
            g.create_dataset(
                "feat",
                data=block,
                dtype="float16",
                chunks=(min(4096, max(1, nrows)), enc.hidden_size),
            )
            g.create_dataset("offsets", data=np.asarray(offs, dtype=np.int64))
            g.create_dataset("lengths", data=np.asarray(lens, dtype=np.int32))
            g.create_dataset("keys", data=np.asarray(keys, dtype=object), dtype=_vlen)
            _fsync_drop(g)
        os.replace(tmp, part_path(ci))

    t0 = time.time()
    last_t, last_done, done, skipped = (t0, 0, 0, 0)
    if rank == 0:
        log.info(
            "encoding %d prompts in %d chunk(s) of %d → %s/part_%d_*.h5 (resumable)",
            len(shard),
            n_chunks,
            chunk_size,
            args.out,
            rank,
        )
    for ci in range(n_chunks):
        lo, hi = (ci * chunk_size, min((ci + 1) * chunk_size, len(shard)))
        c_shard, c_est = (shard[lo:hi], est[lo:hi])
        if chunk_is_done(ci):
            skipped += 1
            done += len(c_shard)
            if rank == 0:
                log.info(
                    "rank0 chunk %d/%d present → skip [resume] (%d prompts)",
                    ci + 1,
                    n_chunks,
                    len(c_shard),
                )
            continue
        arrs, offs, lens, keys, cur = ([], [], [], [], 0)
        for batch in _token_budget_batches(
            c_shard, c_est, args.batch_tokens, args.max_batch
        ):
            seqs = _encode_batch_safe(enc, batch)
            for s, text in zip(seqs, batch):
                a = s.float().cpu().numpy().astype(np.float16)
                arrs.append(a)
                offs.append(cur)
                lens.append(a.shape[0])
                keys.append(sha1(text))
                cur += a.shape[0]
            done += len(batch)
            if rank == 0 and done % 5000 < len(batch):
                now = time.time()
                inst = (done - last_done) / max(1e-09, now - last_t)
                avg = done / max(1e-09, now - t0)
                last_t, last_done = (now, done)
                log.info(
                    "rank0 %d/%d (inst %.0f/s, avg %.0f/s, ~%.0f/s total) chunk=%d/%d",
                    done,
                    len(shard),
                    inst,
                    avg,
                    avg * world,
                    ci + 1,
                    n_chunks,
                )
        block = (
            np.concatenate(arrs, axis=0)
            if arrs
            else np.empty((0, enc.hidden_size), np.float16)
        )
        arrs.clear()
        write_part(ci, block, offs, lens, keys)
        del block, offs, lens, keys
        torch.cuda.empty_cache()
        if rank == 0:
            log.info(
                "rank0 wrote chunk %d/%d → %s (%d prompts, %d tokens)",
                ci + 1,
                n_chunks,
                os.path.basename(part_path(ci)),
                len(c_shard),
                cur,
            )
    log.info(
        "rank%d DONE: %d/%d chunks written (%d skipped-resume), %d prompts, %.0fs → %s/part_%d_*.h5",
        rank,
        n_chunks - skipped,
        n_chunks,
        skipped,
        done,
        time.time() - t0,
        args.out,
        rank,
    )


if __name__ == "__main__":
    main()
