from pathlib import Path
from subprocess import CompletedProcess
import sys


def runtime_err_str(prog: str, cmd: list[str], ret: CompletedProcess[bytes]):
    return (
        f"{prog} returned non-zero status code {ret.returncode}.\n\n"
        f'Full command:\n{"=" * 50}\n{cmd}\n{" ".join(cmd)}\n\n'
        f'stdout:\n{"=" * 50}\n{ret.stdout.decode()}\n\n'
        f'stderr:\n{"=" * 50}\n{ret.stderr.decode()}'
    )

def get_args() -> tuple[Path, Path]:
    if len(sys.argv) != 2:
        print(f"Wrong number of arguments ({len(sys.argv)} given).")
        print("Usage: spoonfy.py <input-video>")
        sys.exit(1)
    in_fn = Path(sys.argv[1])  # e.g. /path/to/file.mp4
    if not in_fn.exists():
        print(f"Input file {str(in_fn)} does not exist!")
        sys.exit(1)
    workdir = in_fn.parent / in_fn.stem  # e.g. /path/to/file.mp4 -> /path/to/file/
    workdir.mkdir(exist_ok=True)
    return in_fn, workdir