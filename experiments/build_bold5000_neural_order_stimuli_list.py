from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a stimuli list in the same order as BOLD5000 GLM beta rows (5254 trials). "
            "Uses official run presentation lists under Stimuli_Presentation_Lists, not a "
            "directory glob (which misses repeats and wrong order)."
        )
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="CSI1",
        help="Subject folder under Stimuli_Presentation_Lists (CSI1, CSI2, ...). Row order is shared across subjects.",
    )
    parser.add_argument(
        "--presentation-root",
        type=Path,
        default=Path(
            "data/raw/bold5000/extracted/BOLD5000_Stimuli/Stimuli_Presentation_Lists"
        ),
        help="Parent of per-subject presentation list folders.",
    )
    parser.add_argument(
        "--presented-stimuli-dir",
        type=Path,
        default=Path(
            "data/raw/bold5000/extracted/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli"
        ),
        help="Root containing COCO/, ImageNet/, Scene/ image files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output text file: one path per line, relative to --presented-stimuli-dir.",
    )
    return parser.parse_args()


def parse_sess_run(path: Path) -> tuple[int, int]:
    match = re.search(r"_sess(\d+)_run(\d+)\.txt$", path.name)
    if not match:
        return (9999, 9999)
    return int(match.group(1)), int(match.group(2))


def collect_ordered_stimulus_names(subject: str, presentation_root: Path) -> list[str]:
    subj_root = (presentation_root / subject).resolve()
    if not subj_root.is_dir():
        raise FileNotFoundError(f"Missing presentation lists directory: {subj_root}")

    run_files = [
        path
        for path in subj_root.rglob("*_sess*_run*.txt")
        if "setup" not in path.name.lower() and path.name.endswith(".txt")
    ]
    run_files = sorted(run_files, key=lambda p: (parse_sess_run(p), str(p)))

    names: list[str] = []
    for path in run_files:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return names


def strip_repeat_prefix(name: str) -> str:
    if name.startswith("rep_"):
        return name[4:]
    return name


def resolve_image_path(name: str, presented: Path) -> Path:
    """Map a presentation-list filename to a file under Presented_Stimuli."""
    base = strip_repeat_prefix(name.strip())
    lower = base.lower()

    if lower.startswith("coco_"):
        candidate = presented / "COCO" / base
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"COCO image not found: {candidate}")

    if re.match(r"^n\d{8}_", base):
        candidate = presented / "ImageNet" / base
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"ImageNet image not found: {candidate}")

    for subdir in ("Scene", "ImageNet", "COCO"):
        candidate = presented / subdir / base
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Could not resolve stimulus '{name}' under {presented}")


def main() -> None:
    args = parse_args()
    presented = args.presented_stimuli_dir.resolve()
    if not presented.is_dir():
        raise FileNotFoundError(f"Presented stimuli directory does not exist: {presented}")

    names = collect_ordered_stimulus_names(args.subject, args.presentation_root.resolve())
    if len(names) != 5254:
        raise ValueError(
            f"Expected 5254 presentation-list lines for BOLD5000 neural RDM alignment; got {len(names)}. "
            f"Check --subject and --presentation-root."
        )

    rel_lines: list[str] = []
    for name in names:
        abs_path = resolve_image_path(name, presented)
        rel_lines.append(abs_path.relative_to(presented).as_posix())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rel_lines) + "\n")
    print(f"Wrote {len(rel_lines)} lines to {args.output} (relative to {presented})")


if __name__ == "__main__":
    main()
