

import subprocess
import time


gpu_command_pair = {
    1: "echo ok 1",
    3: "python -V"
}


POLL_SECONDS = 3


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def _list_gpu_uuids() -> dict[int, str]:
    """Return mapping {gpu_index: gpu_uuid}."""
    output = _run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
    mapping: dict[int, str] = {}
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        mapping[int(parts[0])] = parts[1]
    return mapping


def _pid_username(pid: str) -> str:
    """Return username for pid; empty string if process no longer exists."""
    try:
        return _run(["ps", "-o", "user=", "-p", pid])
    except subprocess.CalledProcessError:
        return ""


def _busy_gpus_non_root() -> set[int]:
    """Return set of GPU indices with at least one non-root process using the GPU."""
    gpu_uuid_map = _list_gpu_uuids()
    uuid_to_index = {uuid: idx for idx, uuid in gpu_uuid_map.items()}

    try:
        output = _run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ]
        )
    except subprocess.CalledProcessError:
        return set()

    busy: set[int] = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        gpu_uuid, pid = parts
        gpu_idx = uuid_to_index.get(gpu_uuid)
        if gpu_idx is None:
            continue

        user = _pid_username(pid)
        if user and user != "root":
            busy.add(gpu_idx)

    return busy


def _launch_in_current_session(gpu_idx: int, command: str) -> None:
    """Run command in the current terminal session and stream output inline."""
    print(f"Running on GPU {gpu_idx} in current session: {command}")
    subprocess.run(["bash", "-lc", command], check=False)


def main() -> None:
    pending = {gpu: cmd for gpu, cmd in gpu_command_pair.items() if cmd.strip()}

    if not pending:
        print("No commands configured in gpu_command_pair; exiting.")
        return

    print(f"Watching GPUs every {POLL_SECONDS}s. Pending GPUs: {sorted(pending)}")

    while pending:
        try:
            busy = _busy_gpus_non_root()
        except FileNotFoundError:
            print("nvidia-smi not found. Exiting.")
            return
        except Exception as exc:
            print(f"Failed to query GPU usage: {exc}")
            time.sleep(POLL_SECONDS)
            continue

        for gpu_idx in list(pending):
            if gpu_idx in busy:
                print(f"GPU {gpu_idx} busy (non-root process detected).")
                continue

            command = pending.pop(gpu_idx)
            print(f"GPU {gpu_idx} appears free. Launching: {command}")

            _launch_in_current_session(gpu_idx, command)

        if pending:
            time.sleep(POLL_SECONDS)

    print("All configured commands have been launched.")


if __name__ == "__main__":
    main()

