"""Names for training eval files, sorted by study, target, sample, then output role."""
from pathlib import Path
import re


MODALITY_NAMES = {0: "t1", 1: "t1c", 2: "t2", 3: "flair", 4: "seg"}
MODE_NAMES = {
    "inpaint": "healthy_region",
    "inpaint_tumor": "tumor_region",
    "inpaint_lesion": "lesion_rebuild",
    "seg_1mod": "refs01", "seg_2mod": "refs02",
    "seg_3mod": "refs03", "seg_4mod": "refs04",
}


def study_key(image_path: str) -> str:
    # Matches MultiTaskDataset's study grouping; includes the complete parent path,
    # so equal accession names under different patient directories stay separate.
    return str(Path(image_path).parent)


def build_eval_case_ids(image_paths) -> dict[str, str]:
    return {key: f"case{i:06d}" for i, key in enumerate(
        sorted({study_key(path) for path in image_paths}), start=1)}


def filename_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value)).strip("-_").lower()
    token = re.sub(r"_+", "_", token)
    return token or "unknown"


def eval_file_base(task: str, mode: str, case_id: str, target_modality: int,
                   save_id: str) -> str:
    """Return a path relative to infer/step_N, with no role suffix or extension."""
    # val2 must sort before val10, and repeated draws of one target must not overwrite.
    sample_id = re.sub(r"\d+", lambda m: f"{int(m[0]):06d}", filename_token(save_id))
    modality = MODALITY_NAMES[int(target_modality)]
    variant = MODE_NAMES.get(mode, filename_token(mode))
    directory = filename_token(task)
    if task == "inpaint":
        directory += f"/{variant}"
    elif mode.endswith("_native_thick"):
        directory += "/native_thick"
    return f"{directory}/{filename_token(case_id)}_{modality}_{sample_id}_{variant}"


def eval_ref_suffix(position: int, count: int, role: str, modality: int,
                    *, decoded: bool) -> str:
    name = filename_token(role)
    if modality in MODALITY_NAMES and modality != 4:
        name += f"_{MODALITY_NAMES[modality]}"
    if decoded:
        name += "_vae_recon"
    return f"_10_ref{position:02d}of{count:02d}_{name}.h5"
