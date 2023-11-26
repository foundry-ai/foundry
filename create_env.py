#!/usr/bin/env python3

import os
os.environ["CONDA_OVERRIDE_CUDA"] = "11.8"
import subprocess
import platform

def create(env_name, pkgs, channels):
    args = ['mamba', 'create', '-y', '-n', env_name, 
         *pkgs]
    for c in channels:
        args.append('-c')
        args.append(c)
    print("Running command:", " ".join(args))

    # delete the env, if it exists
    subprocess.run(
        ['mamba', 'env', 'remove', '-n', env_name]
    )
    subprocess.run(
        args
    )

def run(env_name, cuda=False):
    print(f"Using cuda: {cuda}")
    channels = ['conda-forge']
    # if on satori, add opence channel
    if 'ppc64le' in platform.platform():
        channels.append("https://ftp.osuosl.org/pub/open-ce/current/")
    pkgs = [
        'python==3.10',
        'jax==0.4.7',
        'jaxlib==0.4.7' if not cuda else 'jaxlib==0.4.7[build=cuda118*]',
        'optax==0.1.7',
        'wandb==0.15.12',
        'ffmpeg==6.1.0',
    ]
    if cuda:
        pkgs.append('cuda-toolkit')
        pkgs.append('cuda-nvcc')
        channels.append('nvidia/label/cuda-11.8.0')
    create(env_name, pkgs, channels)

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
