#!/usr/bin/env python3

import os
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
    pkgs = []
    # if on satori, add opence channel
    if 'ppc64le' in platform.platform():
        channels.append("https://packages.pfrommer.dev/opence/")
        pkgs.append('cudnn==8.8.1_12.2')
    pkgs.extend([
        'python==3.11',
        'numpy==1.26.2',
        'jax==0.4.7',
        'jaxlib==0.4.7' if not cuda else 'jaxlib==0.4.7[build=cuda12*]',
        'wandb==0.15.12',
        'matplotlib==3.8.2',
        'ipykernel==6.25.0',
        'ffmpeg==6.1.0',
    ])
    # if on ppc64le, open-ce takes care of cuda
    if cuda:
        pkgs.append('cuda-nvcc')
        channels.append('nvidia/label/cuda-12.2.0')
    create(env_name, pkgs, channels)

    # install mock tensorstore
    subprocess.run(
        ['mamba', 'run', '-n', env_name, 'pip', 'install', 'external/tensorstore-mock']
    )
    # install stanza as editable
    # into the created environment
    print("Installing stanza as editable")
    subprocess.run(
        ['mamba', 'run', '-n', env_name, 'pip', 'install', '--use-pep517', '-e', 'lib']
    )
    # add flax (without any dependencies)

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
