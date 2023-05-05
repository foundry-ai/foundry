import itertools
import gdown
import numpy as np

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
    import gdown
    import os
    tm = np.load(path, allow_pickle=True).item()["params"]
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
    input_path = sys.argv[0]
    output_path = sys.argv[1]
    params = get_parameters(input_path)
    np.save(output_path, params)