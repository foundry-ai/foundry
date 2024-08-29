![Docs](https://github.com/pfrommerd/foundry/actions/workflows/docs.yml/badge.svg)

# [Documentation](https://dan.pfrommer.us/foundry/)

# Installation Instructions

 1. Install Python >= 3.10
 2. Install [PDM](https://pdm-project.org/latest/).
    ```bash
    curl -sSL https://pdm-project.org/install-pdm.py | python3 -
    ```
 3. To set up the foundry environment, run
    ```bash
    pdm install -d
    ```
 4. Then try some notebooks:
    ```bash
    pdm run jupyter lab
    ```
