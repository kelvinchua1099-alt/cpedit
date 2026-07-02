"""
render_grid.py — grid comparison: rows = uid, cols = [source, full, no_crop, GT].
Single canonical view per cell. Textured, orientation-corrected, white bg.

Usage: python scripts/render_grid.py out.png uid_1 uid_2 ...
Reads dataset meshes from data/nano3d/<uid>/ and results from outputs/batch/.
"""
import sys, os
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from PIL import Image, ImageDraw

DEV = "cuda"
H = W = 300
AZ, EL = 25.0, 15.0
FLIP = torch.tensor([1.0, -1.0, -1.0], device=DEV)


def load_mesh(path):
    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values() if hasattr(g, "vertices")])
    v = torch.tensor(np.asarray(m.vertices, np.float32), device=DEV) * FLIP
    f = torch.tensor(np.asarray(m.faces, np.int32), device=DEV)
    v = v - (v.amax(0) + v.amin(0)) / 2.0
    v = v / (v.norm(dim=1).max() + 1e-8)
    uv = tex = None
    vis = getattr(m, "visual", None)
    try:
        if vis is not None and getattr(vis, "uv", None) is not None and vis.material.baseColorTexture is not None:
            uv = torch.tensor(np.asarray(vis.uv, np.float32), device=DEV)
            tex = torch.tensor(np.asarray(vis.material.baseColorTexture.convert("RGB"), np.float32) / 255.0, device=DEV)[None]
    except Exception:
        uv = tex = None
    return v, f, uv, tex


def vnorm(v, f):
    vn = torch.zeros_like(v); tri = v[f]
    fn = torch.nn.functional.normalize(torch.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0], dim=1), dim=1)
    for k in range(3): vn.index_add_(0, f[:, k].long(), fn)
    return torch.nn.functional.normalize(vn, dim=1)


def mvp(az, el, dist=2.6):
    a, e = np.radians(az), np.radians(el)
    eye = np.array([dist*np.cos(e)*np.sin(a), dist*np.sin(e), dist*np.cos(e)*np.cos(a)], np.float32)
    fwd = -eye/np.linalg.norm(eye); up = np.array([0, 1, 0], np.float32)
    s = np.cross(fwd, up); s /= np.linalg.norm(s); u = np.cross(s, fwd)
    V = np.eye(4, dtype=np.float32); V[0, :3], V[1, :3], V[2, :3] = s, u, -fwd; V[:3, 3] = -V[:3, :3] @ eye
    fov, n, fa = np.radians(40), 0.1, 10.0; t = np.tan(fov/2)
    P = np.zeros((4, 4), np.float32); P[0, 0] = 1/t; P[1, 1] = 1/t
    P[2, 2] = -(fa+n)/(fa-n); P[2, 3] = -2*fa*n/(fa-n); P[3, 2] = -1
    return torch.tensor(P @ V, device=DEV)


def render_cell(path, glctx):
    try:
        v, f, uv, tex = load_mesh(path)
    except Exception:
        return np.full((H, W, 3), 240, np.uint8)
    vn = vnorm(v, f)
    m = mvp(AZ, EL)
    clip = (torch.cat([v, torch.ones_like(v[:, :1])], 1) @ m.T)[None]
    rast, _ = dr.rasterize(glctx, clip, f, resolution=[H, W])
    nz = torch.nn.functional.normalize(dr.interpolate(vn[None], rast, f)[0], dim=-1)[..., 2].clamp(0, 1)
    if uv is not None:
        col = dr.texture(tex, dr.interpolate(uv[None], rast, f)[0], filter_mode="linear")
    else:
        col = torch.ones(1, H, W, 3, device=DEV) * 0.6
    shade = (0.45 + 0.55 * nz)[..., None]
    col = col * shade * 0.75 + shade * 0.25
    col = dr.antialias(col.contiguous(), rast, clip, f)
    mask = (rast[..., 3:] > 0).float()
    col = col * mask + (1 - mask)
    return (col[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def main():
    out = sys.argv[1]; uids = sys.argv[2:]
    glctx = dr.RasterizeCudaContext()
    cols = ["edit_512", "source", "full", "no_crop", "GT"]
    grid = []
    for uid in uids:
        paths = {
            "source": f"data/nano3d/{uid}/src_mesh.glb",
            "full":   f"outputs/batch/full/{uid}/result_00.glb",
            "no_crop": f"outputs/batch/no_crop/{uid}/result_00.glb",
            "GT":     f"data/nano3d/{uid}/tar_mesh.glb",
        }
        cells = []
        for c in cols:
            if c == "edit_512":                        # 2D target image, not a mesh
                im = Image.open(f"data/nano3d/{uid}/edit_512.png").convert("RGB").resize((W, H))
                cells.append(np.asarray(im))
            else:
                cells.append(render_cell(paths[c], glctx))
        grid.append(np.concatenate(cells, axis=1))
    img = np.concatenate(grid, axis=0)
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    for j, c in enumerate(cols):
        d.text((j * W + 6, 4), c, fill=(210, 20, 20))
    for i, uid in enumerate(uids):
        d.text((4, i * H + 16), uid, fill=(20, 20, 210))
    im.save(out); print("saved", out, im.size)


if __name__ == "__main__":
    main()
