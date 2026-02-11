import os
import shutil

SRC_ROOT = "/hdd/ivny/results"
NEW_ROOT = "/hdd/ivny/in_domain_univariate_calibration"

FILES_TO_COPY = [
    "hedging_words_cache.pkl",
    "linguistic_calibration_outputs_rewrites.pkl",
    "linguistic_calibration_outputs.csv",
    "linguistic_calibration_outputs.pkl",
]

def collect_results_no_timestamp(src_root=SRC_ROOT, dst_root=NEW_ROOT):
    copied = 0
    skipped = 0

    for dataset in sorted(os.listdir(src_root)):
        dataset_path = os.path.join(src_root, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for calib_method in sorted(os.listdir(dataset_path)):
            calib_path = os.path.join(dataset_path, calib_method)
            if not os.path.isdir(calib_path):
                continue

            for model_family in sorted(os.listdir(calib_path)):
                mf_path = os.path.join(calib_path, model_family)
                if not os.path.isdir(mf_path):
                    continue

                for model_name in sorted(os.listdir(mf_path)):
                    mn_path = os.path.join(mf_path, model_name)
                    if not os.path.isdir(mn_path):
                        continue

                    dst_dir = os.path.join(
                        dst_root,
                        dataset,
                        calib_method,
                        model_family,
                        model_name,
                    )
                    os.makedirs(dst_dir, exist_ok=True)

                    for timestamp in sorted(os.listdir(mn_path)):
                        ts_path = os.path.join(mn_path, timestamp)
                        if not os.path.isdir(ts_path):
                            continue

                        for fname in FILES_TO_COPY:
                            src_file = os.path.join(ts_path, fname)
                            if not os.path.exists(src_file):
                                skipped += 1
                                continue

                            dst_file = os.path.join(
                                dst_dir, f"{fname}"
                            )
                            shutil.copy2(src_file, dst_file)
                            copied += 1

    print(f"[DONE] Copied {copied} files")
    print(f"[INFO] Skipped {skipped} missing files")
    print(f"[ROOT] Output: {dst_root}")


if __name__ == "__main__":
    collect_results_no_timestamp()
