"""
asset_loader.py — render a 3D asset file to a PIL image for TRELLIS 2.0.

TRELLIS 2.0 does not expose a mesh-to-SLAT encoder, so importing a 3D asset
requires rendering it to an image first and using the image-to-3D pipeline.

Rendering backend priority:
  1. pyrender  (fast, GPU/osmesa, best quality)
  2. trimesh built-in  (CPU ray-tracing, slower, no dependencies)

Cloud setup (headless):
  pip install pyrender trimesh[easy]
  apt-get install -y libosmesa6-dev
  export PYOPENGL_PLATFORM=osmesa
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _camera_pose(elevation_deg: float, azimuth_deg: float, distance: float = 3.0) -> np.ndarray:
    """
    Spherical coords → 4×4 camera-to-world (OpenGL convention: -Z forward).

    Args
        elevation_deg : angle above the XZ plane (positive = above)
        azimuth_deg   : angle around Y axis (0 = front / +Z)
        distance      : distance from origin
    """
    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)

    eye = np.array([
        distance * math.cos(el) * math.sin(az),
        distance * math.sin(el),
        distance * math.cos(el) * math.cos(az),
    ])
    target = np.zeros(3)
    world_up = np.array([0.0, 1.0, 0.0])

    fwd   = target - eye;  fwd   /= np.linalg.norm(fwd)
    right = np.cross(fwd, world_up);  right /= np.linalg.norm(right)
    up    = np.cross(right, fwd)

    pose = np.eye(4)
    pose[:3, 0] =  right
    pose[:3, 1] =  up
    pose[:3, 2] = -fwd   # OpenGL: -Z looks toward target
    pose[:3, 3] =  eye
    return pose


# ---------------------------------------------------------------------------
# Mesh loading + normalisation
# ---------------------------------------------------------------------------

def _load_mesh(asset_path: str) -> "trimesh.Trimesh":
    """
    Load a 3D asset file and return a single normalised Trimesh.

    Supports: .glb .gltf .obj .ply .stl .off
    The mesh is centred at the origin and scaled to fit inside a unit sphere.
    """
    import trimesh

    obj = trimesh.load(asset_path, force="scene", process=False)

    if isinstance(obj, trimesh.Trimesh):
        mesh = obj.copy()
    else:
        # Scene: merge all geometries, applying their transforms
        mesh = obj.dump(concatenate=True)

    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError(f"No geometry found in {asset_path!r}")

    # Centre and scale to unit sphere
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.apply_translation(-center)
    scale = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    if scale > 1e-6:
        mesh.apply_scale(2.0 / scale)

    return mesh


# ---------------------------------------------------------------------------
# pyrender backend
# ---------------------------------------------------------------------------

def _render_pyrender(
    mesh:      "trimesh.Trimesh",
    size:      int,
    bg_color:  Tuple[int, int, int],
    cam_pose:  np.ndarray,
) -> Image.Image:
    import pyrender
    import numpy as np

    bg = [c / 255.0 for c in bg_color] + [1.0]
    scene = pyrender.Scene(bg_color=bg, ambient_light=[0.25, 0.25, 0.25])

    pymesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(pymesh)

    cam = pyrender.PerspectiveCamera(yfov=math.pi / 3.0, aspectRatio=1.0)
    scene.add(cam, pose=cam_pose)

    # Key light (tracks camera)
    key = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    scene.add(key, pose=cam_pose)

    # Fixed fill light from above-front
    fill_pose = np.eye(4)
    fill_pose[:3, 3] = [0.5, 2.0, 1.5]
    fill = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.5)
    scene.add(fill, pose=fill_pose)

    r = pyrender.OffscreenRenderer(viewport_width=size, viewport_height=size)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()

    # Alpha-composite onto bg_color
    alpha = color[:, :, 3:4] / 255.0
    bg_arr = np.array(bg_color, dtype=np.uint8)
    rgb = (color[:, :, :3] * alpha + bg_arr * (1.0 - alpha)).astype(np.uint8)
    return Image.fromarray(rgb)


# ---------------------------------------------------------------------------
# trimesh built-in backend (CPU ray-tracing)
# ---------------------------------------------------------------------------

def _render_trimesh(
    mesh:      "trimesh.Trimesh",
    size:      int,
    bg_color:  Tuple[int, int, int],
    cam_pose:  np.ndarray,
) -> Image.Image:
    """
    CPU ray-cast render via trimesh.  Much slower but zero extra dependencies.
    Produces flat-shaded output without lighting.
    """
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    from trimesh.scene.cameras import Camera

    scene = trimesh.scene.Scene([mesh])
    scene.camera = Camera(resolution=(size, size), fov=(60, 60))
    scene.camera_transform = cam_pose

    try:
        img_bytes = scene.save_image(resolution=(size, size), background=bg_color + (255,))
        return Image.open(__import__("io").BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            f"trimesh built-in renderer failed: {exc}\n"
            "Install pyrender for better results: pip install pyrender"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_asset(
    asset_path:    str,
    size:          int   = 512,
    elevation_deg: float = 15.0,
    azimuth_deg:   float = 30.0,
    bg_color:      Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Load a 3D asset and render it to a PIL image from a canonical viewpoint.

    The rendered image is suitable for use as img_src in the VS3D pipeline.
    A slight elevation (15°) and azimuth (30°) gives a 3/4 view that shows
    the main visual features without losing the top surface.

    Args
    ----
    asset_path    : path to .glb / .gltf / .obj / .ply / .stl
    size          : output image size in pixels (square)
    elevation_deg : camera elevation above horizontal plane
    azimuth_deg   : camera rotation around vertical axis (0 = front)
    bg_color      : background RGB tuple (default white)

    Returns
    -------
    PIL.Image  (RGB, size×size)

    Notes
    -----
    Cloud headless setup:
        pip install pyrender trimesh[easy]
        apt-get install -y libosmesa6-dev
        export PYOPENGL_PLATFORM=osmesa
    """
    mesh     = _load_mesh(asset_path)
    cam_pose = _camera_pose(elevation_deg, azimuth_deg)

    try:
        import pyrender  # noqa: F401
        return _render_pyrender(mesh, size, bg_color, cam_pose)
    except ImportError:
        pass

    return _render_trimesh(mesh, size, bg_color, cam_pose)
