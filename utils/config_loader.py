from __future__ import annotations
import os
import re
import sys
import argparse
from typing import Any
import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in {"task_weights", "cfg_train_states", "task_cfg", "eval_tasks"}:
            out[key] = value
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _expand_environment(value):
    if isinstance(value, dict):
        return {k: _expand_environment(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_environment(v) for v in value]
    if isinstance(value, str) and "${" in value:
        os.environ.setdefault("HF_HOME", "/working/huggingface_cache")
        missing = [
            key
            for key in re.findall(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", value)
            if not os.environ.get(key)
        ]
        if missing:
            raise ValueError(
                "Set required environment variable(s): " + ", ".join(missing)
            )
        return os.path.expandvars(value)
    return value


def normalize_infer_mode_names(cfg: dict[str, Any]) -> dict[str, Any]:
    """Accept archived eval names only at the config boundary; emit canonical names."""
    infer = cfg.get("infer") or {}
    for field in ("task_cfg", "eval_tasks"):
        modes = infer.get(field) or {}
        if "no_ref" in modes:
            if "modality_only" in modes and modes["modality_only"] != modes["no_ref"]:
                raise ValueError(f"Conflicting infer.{field} entries: no_ref and modality_only")
            modes["modality_only"] = modes.pop("no_ref")
    if infer.get("best_fid_mode") == "no_ref":
        infer["best_fid_mode"] = "modality_only"
    return cfg


def load_train_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = _expand_environment(yaml.safe_load(f))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {config_path} must be a mapping at the top level.")
    normalize_infer_mode_names(cfg)
    base_path = cfg.pop("base_config", None)
    if base_path:
        if not os.path.isabs(base_path):
            base_path = os.path.join(
                os.path.dirname(os.path.abspath(config_path)), base_path
            )
        base = load_train_config(base_path)
        base.pop("__config_path__", None)
        base.pop("__config_dir__", None)
        cfg = _deep_merge(base, cfg)
    cfg["__config_path__"] = os.path.abspath(config_path)
    cfg["__config_dir__"] = os.path.dirname(cfg["__config_path__"])
    return cfg


def parse_yaml_args(parser: argparse.ArgumentParser, section: str, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv:
        return parser.parse_args(argv)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", default=parser.get_default("config"))
    selected, _ = probe.parse_known_args(argv)
    cfg = load_train_config(selected.config)
    defaults = cfg.get(section, {})
    if not isinstance(defaults, dict):
        parser.error(f"{section} must be a YAML mapping")
    actions = {a.dest: a for a in parser._actions if a.dest not in {"help", "config"}}
    for key, value in defaults.items():
        if key not in actions:
            parser.error(f"unknown YAML parameter {section}.{key}")
        action = actions[key]
        if value is not None:
            is_list = action.nargs in ("*", "+") or (
                isinstance(action.nargs, int) and action.nargs > 0
            )
            if is_list and (not isinstance(value, list)):
                parser.error(f"{section}.{key} must be a YAML list")
            values = value if is_list else [value]
            try:
                values = [action.type(v) if action.type else v for v in values]
            except (ValueError, TypeError) as exc:
                parser.error(f"{section}.{key}: {exc}")
            if action.choices is not None and any(
                (v not in action.choices for v in values)
            ):
                parser.error(f"{section}.{key} must use choices {list(action.choices)}")
            if isinstance(
                action,
                (
                    argparse._StoreTrueAction,
                    argparse._StoreFalseAction,
                    argparse.BooleanOptionalAction,
                ),
            ) and (not isinstance(value, bool)):
                parser.error(f"{section}.{key} must be a YAML boolean")
            value = values if is_list else values[0]
            action.required = False
        parser.set_defaults(**{key: value})
    args = parser.parse_args(argv)
    args._config = cfg
    return args


def ensure_project_root_on_path() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def resolve_path(cfg: dict[str, Any], path: str) -> str:
    if os.path.isabs(path):
        return path
    base = cfg.get("__config_dir__", os.getcwd())
    return os.path.normpath(os.path.join(base, path))
