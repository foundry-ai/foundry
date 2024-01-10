import inquirer
from rich import print
import sys
import platform
from pathlib import Path

arch = platform.machine()
# remap
if arch == "aarch64": arch = "arm64"
if arch == "x86_64": arch = "amd64"

config = {
  "context": ["requirements.txt"],
  "shell": "fish",
  "arch": arch,
  "cuda": "12.2" if sys.platform == 'linux' else "none",
  "build": []
}

# custom package options
custom_packages = [p.suffix.lstrip(".")
  for p in (Path(__file__).parent / "custom").iterdir()
]

use_defaults = inquirer.prompt([
  inquirer.Confirm("defaults", 
    message="Do you want to use the default configuration?",
    default=True
  )
])["defaults"]

if not use_defaults:
  config.update(inquirer.prompt([
    inquirer.List('shell',
                  message="What shell do you want to use?",
                  choices=['fish', 'bash'],
                  default=config['shell']
    ),
    inquirer.List('arch',
      message="What architecture should we build for?",
      choices=["amd64", "arm64", "ppc64le"],
      default=arch
    ),
    inquirer.List("cuda",
      message="What cuda version do you want to use?",
      choices=["12.2", "none"],
      default=(None if sys.platform == 'linux' else 'none')
    ),
    inquirer.Checkbox("build",
      message="What custom packages do you want to build from source?",
      choices=custom_packages,
    )
  ]))