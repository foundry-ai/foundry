import shutil
import pathlib
from typing import Optional

BINARY = "@binary_path@"

def build_wandb_core(
    go_binary: pathlib.Path,
    output_path: pathlib.PurePath,
    with_code_coverage: bool,
    with_race_detection: bool,
    wandb_commit_sha: Optional[str],
    target_system,
    target_arch,
) -> None:
    print(f"Copying wandb-core binary {BINARY}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BINARY, output_path)
    output_path.chmod(0o755)