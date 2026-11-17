"""legacy 实现 vs 向量化实现的数值等价性检查

用法:
    python tools/check_equiv.py            # 默认 CPU
    python tools/check_equiv.py --device cuda
"""
import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.maptr import BEVTransform
from models.bevformer.encoder import point_sampling, point_sampling_vectorized
from models.bevformer.spatial_cross_attention import (
    SpatialCrossAttention, MSDeformableAttention3D,
)


def _allclose(a, b, name, atol=1e-5, rtol=1e-4):
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    diff = (a - b).abs().max().item() if a.numel() else 0.0
    print(f'  [{"OK" if ok else "FAIL"}] {name}  max_diff={diff:.3e}')
    return ok


def test_point_sampling(device):
    print('\n== point_sampling ==')
    torch.manual_seed(0)
    B, C, D, N = 2, 6, 4, 64
    pc_range = [-10, -10, -3, 30, 10, 5]
    img_h, img_w = 176, 320

    ref_3d = torch.rand(B, D, N, 3, device=device)
    intrinsics = torch.eye(3, device=device).repeat(B, C, 1, 1)
    intrinsics[..., 0, 0] = 120.0
    intrinsics[..., 1, 1] = 120.0
    intrinsics[..., 0, 2] = 160.0
    intrinsics[..., 1, 2] = 88.0
    extrinsics = torch.eye(4, device=device).repeat(B, C, 1, 1)
    extrinsics[..., 0, 3] = torch.randn(B, C, device=device) * 2
    extrinsics[..., 1, 3] = torch.randn(B, C, device=device) * 2
    # 随机小角度旋转
    ang = torch.randn(B, C, device=device) * 0.2
    c, s = torch.cos(ang), torch.sin(ang)
    extrinsics[..., 0, 0], extrinsics[..., 0, 2] = c, s
    extrinsics[..., 2, 0], extrinsics[..., 2, 2] = -s, c

    ref_l, mask_l = point_sampling(ref_3d, pc_range, intrinsics, extrinsics, img_h, img_w)
    ref_v, mask_v = point_sampling_vectorized(ref_3d, pc_range, intrinsics, extrinsics, img_h, img_w)
    same_mask = bool((mask_v == mask_l).all().item())
    if same_mask and mask_l.any():
        ok = _allclose(ref_v[mask_v], ref_l[mask_l], 'reference_points (valid only)')
    else:
        ok = _allclose(ref_v, ref_l, 'reference_points')
    ok &= same_mask
    print(f'  [{"OK" if same_mask else "FAIL"}] bev_mask identical')
    return ok


def test_spatial_cross_attention(device):
    print('\n== SpatialCrossAttention (vectorized vs legacy) ==')
    torch.manual_seed(1)
    B, C, D, Nq = 2, 3, 2, 12
    H, W = 4, 6
    L = H * W
    embed_dims, heads, num_points = 16, 4, 2

    deform_attn = MSDeformableAttention3D(
        embed_dims=embed_dims, num_heads=heads, num_levels=1, num_points=num_points,
        dropout=0.0, batch_first=True)
    attn = SpatialCrossAttention(
        embed_dims=embed_dims, num_cams=C, dropout=0.0, batch_first=True,
        deformable_attention=deform_attn, impl='legacy').to(device)
    attn.eval()

    query = torch.randn(B, Nq, embed_dims, device=device)
    value = torch.randn(C, L, B, embed_dims, device=device)
    key = value.clone()

    # 故意构造不均衡可见数: cam0=8, cam1=2, cam2=0 (空相机)
    mask = torch.zeros(C, B, Nq, D, device=device, dtype=torch.bool)
    mask[0, :, :8, :] = True
    mask[1, :, :2, :] = True
    ref = torch.rand(C, B, Nq, D, 2, device=device)
    spatial_shapes = torch.tensor([[H, W]], dtype=torch.long, device=device)
    level_start_index = torch.tensor([0], dtype=torch.long, device=device)

    def run(impl, grouping, bucket):
        attn.impl = impl
        attn.cam_grouping = grouping
        attn.len_bucket = bucket
        return attn(query, key, value, reference_points_cam=ref, bev_mask=mask,
                    spatial_shapes=spatial_shapes, level_start_index=level_start_index)

    with torch.no_grad():
        out_legacy = run('legacy', True, 0)
        out_group = run('vectorized', True, 0)
        out_padded = run('vectorized', False, 0)
        out_group_bucket = run('vectorized', True, 8)
        out_padded_bucket = run('vectorized', False, 8)

    ok = _allclose(out_group, out_legacy, 'vectorized+grouping (unbalanced)')
    ok &= _allclose(out_padded, out_legacy, 'vectorized+padded (unbalanced)')
    ok &= _allclose(out_group_bucket, out_legacy, 'grouping + len_bucket=8')
    ok &= _allclose(out_padded_bucket, out_legacy, 'padded + len_bucket=8')
    return ok


def _legacy_transform(bt, bev, flip, rot, dx, dy):
    """复刻改造前的 BEVTransform 逻辑 (用于等价性对照)"""
    B = bev.shape[0]
    if flip is not None:
        mask = torch.tensor(list(flip), device=bev.device, dtype=torch.bool)
        if mask.any():
            bev = bev.clone()
            bev[mask] = bev[mask].flip(dims=[-2])
    need = (
        (rot is not None and torch.tensor(list(rot)).abs().max() > 1e-6) or
        (dx is not None and torch.tensor(list(dx)).abs().max() > 1e-6) or
        (dy is not None and torch.tensor(list(dy)).abs().max() > 1e-6)
    )
    if need:
        device, dtype = bev.device, bev.dtype
        if rot is not None:
            rot_t = torch.tensor(list(rot), device=device, dtype=dtype)
            c, s = torch.cos(rot_t), torch.sin(rot_t)
        else:
            c, s = 1.0, 0.0
        tx = (torch.tensor(list(dx), device=device, dtype=dtype) / bt.half_x) if dx is not None else 0.0
        ty = (torch.tensor(list(dy), device=device, dtype=dtype) / bt.half_y) if dy is not None else 0.0
        theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = c
        theta[:, 0, 1] = -s * bt.ratio_yx
        theta[:, 0, 2] = ((c - 1) * bt.x_center + s * bt.y_center) / bt.half_x - tx
        theta[:, 1, 0] = s * bt.ratio_xy
        theta[:, 1, 1] = c
        theta[:, 1, 2] = (s * bt.x_center + (1 - c) * bt.y_center) / bt.half_y + ty
        grid = torch.nn.functional.affine_grid(theta, bev.shape, align_corners=False)
        bev = torch.nn.functional.grid_sample(
            bev, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return bev


def test_bev_transform(device):
    print('\n== BEVTransform ==')
    torch.manual_seed(2)
    bt = BEVTransform(x_range=40.0, y_range=20.0, x_center=10.0, y_center=0.0)
    bev = torch.randn(4, 8, 10, 20, device=device)

    cases = [
        dict(flip=[False, False, False, False], rot_angle=[0.0] * 4, dx=[0.0] * 4, dy=[0.0] * 4),
        dict(flip=[True, False, True, False], rot_angle=[0.0] * 4, dx=[0.0] * 4, dy=[0.0] * 4),
        dict(flip=[False] * 4, rot_angle=[0.1, -0.2, 0.0, 0.3], dx=[0.0] * 4, dy=[0.0] * 4),
        dict(flip=[True, False, True, False], rot_angle=[0.1, 0.0, -0.1, 0.2],
             dx=[1.0, -1.0, 0.0, 2.0], dy=[0.5, 0.0, -1.5, 1.0]),
    ]
    ok = True
    for i, kw in enumerate(cases):
        with torch.no_grad():
            out_new = bt.transform_bev(bev, **kw)
            out_old = _legacy_transform(bt, bev, kw['flip'], kw['rot_angle'], kw['dx'], kw['dy'])
        ok &= _allclose(out_new, out_old, f'case {i}')

    # tensor 兜底输入应等价于 list 输入
    with torch.no_grad():
        kw = cases[-1]
        out_list = bt.transform_bev(bev, **kw)
        out_tensor = bt.transform_bev(
            bev,
            flip=torch.tensor(kw['flip'], device=device),
            rot_angle=torch.tensor(kw['rot_angle'], device=device),
            dx=torch.tensor(kw['dx'], device=device),
            dy=torch.tensor(kw['dy'], device=device),
        )
    ok &= _allclose(out_tensor, out_list, 'tensor -> list fallback')
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f'device={device}')

    ok = True
    ok &= test_point_sampling(device)
    ok &= test_spatial_cross_attention(device)
    ok &= test_bev_transform(device)
    print('\n===== 等价性检查:', 'PASS' if ok else 'FAIL', '=====')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
