import pathlib
import shutil

class NvidiaGpuStatsBuildError(Exception):
    """Raised when building Nvidia GPU stats fails."""

BINARY = "@binary_path@"

def build_nvidia_gpu_stats(
    cargo_binary: pathlib.Path,
    output_path: pathlib.PurePath,
) -> None:
    print(f"Copying nvidia_gpu_stats binary {BINARY}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BINARY, output_path)
    output_path.chmod(0o755)