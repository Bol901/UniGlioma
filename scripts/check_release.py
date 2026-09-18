"""Screen the source tree before publication; does not publish or contact a service."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    "README.md",
    "requirements.txt",
    "requirements-dev.txt",
    ".gitignore",
    "train.py",
    "infer.py",
}
SOURCE_DIRS = {"models", "data", "utils", "scripts", "tests"}
OTHER_FILES = {
    "configs/train.yaml",
    "configs/infer.yaml",
    "docs/DATA_FORMAT.md",
    "assets/seg_prompts.json",
    "examples/records.example.json",
    "examples/acquisition.example.json",
}
PATTERNS = [
    re.compile(r"/(?:mnt|home)/[A-Za-z0-9_]"),
    re.compile(r"/working/(?!huggingface_cache(?:/|\b))[A-Za-z0-9_]"),
    re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
]


def audit(root=ROOT):
    problems, count = [], 0
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if ".git" in rel.parts:
            continue
        if path.is_symlink():
            problems.append(f"Symlink is not permitted: {rel}")
            continue
        if path.is_dir():
            continue
        name = rel.as_posix()
        allowed = (
            name in ROOT_FILES
            or name in OTHER_FILES
            or (rel.parts[0] in SOURCE_DIRS and path.suffix == ".py")
        )
        if not allowed:
            problems.append(f"Unexpected release file: {name}")
            continue
        count += 1
        if path.stat().st_size > 1_000_000:
            problems.append(f"Unexpected large file: {name}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            problems.append(f"Non-text file: {name}")
            continue
        for pattern in PATTERNS:
            if pattern.search(text):
                problems.append(f"Private path or credential pattern in {name}")
                break
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=name)
            except SyntaxError as exc:
                problems.append(f"Python syntax error: {name}:{exc.lineno}")
    return count, problems


if __name__ == "__main__":
    count, problems = audit()
    if problems:
        raise SystemExit("\n".join(problems))
    print(
        f"PASS: {count} allowlisted text files; no weight/data binaries, symlinks, or detected private-path/credential patterns."
    )
    print(
        "This is a structural screen, not a guarantee that later user-added content is safe to publish."
    )
