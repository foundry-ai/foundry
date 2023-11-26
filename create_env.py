import os
os.environ["CONDA_OVERRIDE_CUDA"] = "11.8"
from mamba.api import create, install
import subprocess

def run(env_name, cuda=False):
    print(f"Using cuda: {cuda}")
    pkgs = [
        'python==3.11',
        'jax==0.4.19',
        'jaxlib==0.4.19',
        'flax==0.7.5',
        'optax==0.1.7',
        'wandb==0.15.12',
        'ffmpeg==6.1.0',
    ]
    if cuda:
        pkgs.append('cuda-toolkit')
        pkgs.append('cuda-nvcc')
    create(env_name,
            pkgs,
           ('conda-forge', 'nvidia/label/cuda-11.8.0')
    )
    if cuda:
        # for whatever reason we need to install jaxlib
        # separately so that mamba realizes that cuda is available
        subprocess.run(
            ['mamba', 'run', '-n', env_name, 'mamba', 'install', 'jaxlib==0.4.19[build=cuda118*]', '-c', 'conda-forge']
        )

    # install stanza as editable
    # into the created environment
    print("Installing stanza as editable")
    subprocess.run(
        ['mamba', 'run', '-n', env_name, 'pip', 'install', '--use-pep517', '-e', 'lib']
    )

if __name__=='__main__':
    import argparse
    import os
    import sys
    cuda_default = sys.platform == 'linux'

    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', default='stanza', nargs='?')
    parser.add_argument('--cuda', action='store_true', dest='cuda', default=cuda_default)
    parser.add_argument('--no-cuda', action='store_false', dest='cuda')
    args = parser.parse_args()
    run(args.env_name, cuda=args.cuda)
