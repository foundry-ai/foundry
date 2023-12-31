import inquirer
from rich import print
import sys
import platform


arch = platform.machine()
# map amd64 to aaarch64
arch = arch if arch != "arm64" else "aarch64"

config = {
  "shell": "fish",
  "arch": arch,
  "cuda": "12.2" if sys.platform == 'linux' else "none",
  "custom": {
    "numpy": {"version": "1.26.2", "build": False},
    "scipy": {"version": "1.11.4", "build": False},
    "jax": {"version": "0.4.23", "build": False},
    "jaxlib": {"version": "0.4.23", "build": False},
    "pandas": {"version": "2.1.4", "build": False},
    "plotly": {"version": "5.18.0", "build": False},
    "matplotlib": {"version": "3.8.2", "build": False},
  }
}

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
      choices=["x86_64", "aarch64", "ppc64le"],
      default=arch
    ),
    inquirer.List("cuda",
      message="What cuda version do you want to use?",
      choices=["12.2", "none"],
      default=(None if sys.platform == 'linux' else 'none')
    ),
  ]))
  packages = set(inquirer.prompt([
    inquirer.Checkbox("custom",
      message="What custom packages do you want to install?",
      choices=[p for p in config["custom"]],
      default=[p for p in config["custom"] if config["custom"][p]["build"]],
    )
  ])["custom"])
  for p in config["custom"]:
    config["custom"][p]["build"] = p in packages