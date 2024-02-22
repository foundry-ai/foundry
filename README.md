# Installation Instructions

 1. Install Python >= 3.10
 2. Install [PDM](https://pdm-project.org/latest/).
    ```bash
    curl -sSL https://pdm-project.org/install-pdm.py | python3 -
    ```
 3. For cpu only support, run
    ```bash
    pdm sync -d --only-keep
    ```
    For cuda support, run
    ```bash
    pdm sync -G cuda12_pip -d --only-keep
    ```
    for cuda support without installing cuda through pip, run
    ```bash
    pdm sync -G cuda12_local -d --only-keep
    ```
 4. Then try some example scripts:
    ```bash
    pdm run python examples/train.py
    ```
