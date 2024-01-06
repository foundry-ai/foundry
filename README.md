# Installation Instructions

 1. Install [Docker Desktop][https://docs.docker.com/engine/install/] (or Docker Engine on Linux)
 2. For CUDA support on linux, install [nvidia container toolkit][https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt]. Once installed, be sure to run the Docker configuration command
    ```bash
        sudo nvidia-ctk runtime configure --runtime=docker
    ```
    and either restart your system or the docker daemon.
 3. Allow local user to run docker without needing sudo:
    ```bash
        usermod -a -G docker $USER
    ```
    Then log out and log back in (or simply restart).
 4. Install denvtool (ensure the python version is >= 3.10)
    ```bash
        pip install git+https://github.com/pfrommerd/denvtool.git
    ```
 5. Start the container. Run the following in the top-level directory:
    ```bash
        denvtool start
    ```
    This will build the environment (which may take a while).

To access a shell in the running container, run ```denvtool shell```.

To connect from VSCode, install the "Dev Containers" extension and then find "Attach to Running Container" from the Ctrl+Shift+P command menu. Then open the folder ```/home/$USER/code```, which is mapped to the stanza folder outside of the container.