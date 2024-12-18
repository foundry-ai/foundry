import itertools
import sys
import numpy as np
from foundry.datasets.util import download, cache_path
import torch
import dill

def download_checkpoint(name, quiet=False):
    """
    Load pretrained weights for a given Diffusion Policy training run.

    Args:
        name (str): whatever appears between experiments/... and .../checkpoints
        max_trajectories (int): maximum number of trajectories to load
        quiet (bool): whether to suppress download progress
    
    Returns:
        env_name (str): name of environment
        data (SequenceData): sequence data containing states and actions for all trajectories
    """
    url = f"https://diffusion-policy.cs.columbia.edu/data/experiments/{name}/checkpoints/latest.ckpt"
    path = cache_path(name, "latest.ckpt")
    download(path,
        job_name=name,
        url=url, 
        quiet=quiet
    )
    return path

def remap_diff_step_encoder():
    yield 'diffusion_step_encoder.1', "net/diff_embed_linear_0"
    yield 'diffusion_step_encoder.3', "net/diff_embed_linear_1"

def remap_resblock(in_prefix, out_prefix):
    for l in [0,1]:
        yield f"{in_prefix}.blocks.{l}.block.0", f"{out_prefix}/block{l}/conv"
        yield f"{in_prefix}.blocks.{l}.block.1", f"{out_prefix}/block{l}/group_norm"
    yield f'{in_prefix}.residual_conv', f'{out_prefix}/residual_conv'
    yield f'{in_prefix}.cond_encoder.1', f'{out_prefix}/cond_encoder'
def remap_downsample(in_prefix, out_prefix):
    yield f'{in_prefix}.conv', f'{out_prefix}/conv'

def remap_upsample(in_prefix, out_prefix):
    yield f'{in_prefix}.conv', f'{out_prefix}/conv_transpose'

MOD_NAME_MAP = dict(itertools.chain(
    remap_diff_step_encoder(),
    remap_resblock('mid_modules.0', 'net/mid0'),
    remap_resblock('mid_modules.1', 'net/mid1'),

    remap_resblock('down_modules.0.0', 'net/down0_res0'),
    remap_resblock('down_modules.0.1', 'net/down0_res1'),
    remap_downsample('down_modules.0.2', 'net/down0_downsample'),
    remap_resblock('down_modules.1.0', 'net/down1_res0'),
    remap_resblock('down_modules.1.1', 'net/down1_res1'),
    remap_downsample('down_modules.1.2', 'net/down1_downsample'),
    remap_resblock('down_modules.2.0', 'net/down2_res0'),
    remap_resblock('down_modules.2.1', 'net/down2_res1'),

    remap_resblock('up_modules.0.0', 'net/up0_res0'),
    remap_resblock('up_modules.0.1', 'net/up0_res1'),
    remap_upsample('up_modules.0.2', 'net/up0_upsample'),
    remap_resblock('up_modules.1.0', 'net/up1_res0'),
    remap_resblock('up_modules.1.1', 'net/up1_res1'),
    remap_upsample('up_modules.1.2', 'net/up1_upsample'),
    [('final_conv.0.block.0', 'net/final_conv_block/conv'),
     ('final_conv.0.block.1', 'net/final_conv_block/group_norm'),
     ('final_conv.1', 'net/final_conv')],
))

def get_parameters(path):
    import os
    payload = torch.load(open(path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    tm = payload['state_dicts']['ema_model']

    mapped_params = {}
    for (k,v) in tm.items():
        v = np.array(v).T
        if k.endswith('.weight'):
            root = k[:-len('.weight')]
            # root = MOD_NAME_MAP[root]
            ext = 'scale' if 'group_norm' in root else 'w'
        elif k.endswith('.bias'):
            root = k[:-len('.bias')]
            ext = 'offset' if 'group_norm' in root else 'b'
        else:
            continue
        if root.startswith('model.'):
            root = root[len('model.'):]
        mapped_root = MOD_NAME_MAP[root]
        if 'group_norm' in mapped_root:
            ext = 'offset' if ext == 'b' else 'scale'
        # print(f'{k} -> {mapped_root} {ext} {v.shape}')
        # Map the root name
        if 'transpose' in mapped_root and ext == 'w':
            # for some reason the conv transposed
            # needs the kernel to be flipped but the
            # regular conv does not?
            v = np.flip(v, 0)
        d = mapped_params.setdefault(mapped_root,{})
        d[ext] = v
    return mapped_params

if __name__ == "__main__":
    # save the parameters:
    name = "low_dim/can_ph/diffusion_policy_cnn/train_0"
    input_path = download_checkpoint(name)
    print(input_path)
    output_path = cache_path(name, "params.npy")
    params = get_parameters(input_path)
    import jax
    params_shape = jax.tree_util.tree_map(lambda x: x.shape, params)
    for (k,v) in params_shape.items():
        print(k, v)
    with open(output_path, 'wb') as f:
        np.save(f, params)