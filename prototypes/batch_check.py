#!/usr/bin/env python3
"""
Check which leaf dirs under RESULTS_ROOT have required output files.

For each target model dir, checks both the source timestamp dir and the
sibling 'retrieved/' dir for task.log and filtered_outputs_0.pkl.

Usage:
    conda run -n llm python prototypes/batch_check.py
"""

from pathlib import Path

RESULTS_ROOT = Path("/hdd/ivny/results")
TASK_PATTERNS = ["direct_qa_unified_lc", "hedged_qa_unified_lc"]
TARGET_MODELS = {
    "gemma-4-31B-it",
    # "Llama-3.3-70B-Instruct-Turbo",
    "gpt-oss-120b",
    "Qwen3-235B-A22B-Instruct-2507-tput",
}

REQUIRED_FILES = ["task.log", "filtered_outputs_0.pkl"]


def check_dir(d: Path) -> dict[str, bool]:
    return {f: (d / f).exists() for f in REQUIRED_FILES}


def find_leaf_dirs() -> list[Path]:
    dirs = []
    for dataset_dir in sorted(RESULTS_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for task_pattern in TASK_PATTERNS:
            task_dir = dataset_dir / task_pattern
            if not task_dir.exists():
                continue
            for model_fam_dir in sorted(task_dir.iterdir()):
                if not model_fam_dir.is_dir():
                    continue
                for model_name_dir in sorted(model_fam_dir.iterdir()):
                    if not model_name_dir.is_dir():
                        continue
                    if model_name_dir.name not in TARGET_MODELS:
                        continue
                    candidates = sorted(
                        d for d in model_name_dir.iterdir()
                        if d.is_dir() and d.name == "retrieved"
                    )
                    if candidates:
                        dirs.append(candidates[-1])
    return dirs


def main() -> None:
    leaf_dirs = find_leaf_dirs()
    if not leaf_dirs:
        print("No matching leaf dirs found.")
        return

    ok = missing = 0
    for source_dir in leaf_dirs:
        retrieved_dir = source_dir.parent / "retrieved"
        src_state = check_dir(source_dir)
        ret_state = check_dir(retrieved_dir) if retrieved_dir.exists() else {f: False for f in REQUIRED_FILES}

        # Summarise
        src_ok = all(src_state.values())
        ret_ok = all(ret_state.values())

        src_tag = "OK" if src_ok else "MISSING:" + ",".join(f for f, v in src_state.items() if not v)
        ret_tag = "OK" if ret_ok else ("NO DIR" if not retrieved_dir.exists() else
                                        "MISSING:" + ",".join(f for f, v in ret_state.items() if not v))

        status = "READY" if ret_ok else "INCOMPLETE"
        if ret_ok:
            ok += 1
        else:
            missing += 1

        print(f"[{status:10s}]  {source_dir}")
        print(f"             src/      : {src_tag}")
        print(f"             retrieved/: {ret_tag}")
        print()

    print(f"Summary: {ok} ready, {missing} incomplete  (out of {len(leaf_dirs)} dirs)")


if __name__ == "__main__":
    main()
