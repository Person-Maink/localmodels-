import cv2
import numpy as np
import torch
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.0):
    """Convert weak-perspective camera from crop coordinates to full-image translation."""
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


def render_rgba_multiple(
    vertices,
    faces_in,
    cam_t,
    render_res,
    focal_length,
    is_right=None,
    mesh_base_color=(0.25098039, 0.274117647, 0.65882353),
    scene_bg_color=(1.0, 1.0, 1.0),
    device=None,
):
    """Render multiple hand meshes to a single RGBA image with PyTorch3D."""
    if len(vertices) == 0:
        raise ValueError("No vertices provided for rendering.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    width, height = int(render_res[0]), int(render_res[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid render resolution: {render_res}")

    if is_right is None:
        is_right = [1] * len(vertices)

    if torch.is_tensor(faces_in):
        faces_right = faces_in.detach().cpu().numpy().copy()
    else:
        faces_right = np.asarray(faces_in, dtype=np.int64).copy()
    faces_left = faces_right[:, [0, 2, 1]]

    mesh_list = []
    for verts_np, cam_np, right_flag in zip(vertices, cam_t, is_right):
        verts_np = np.asarray(verts_np, dtype=np.float32)
        cam_np = np.asarray(cam_np, dtype=np.float32)
        faces_np = faces_right if int(right_flag) == 1 else faces_left

        verts = torch.as_tensor(verts_np, dtype=torch.float32, device=device)
        verts = verts + torch.as_tensor(cam_np, dtype=torch.float32, device=device)[None, :]
        # Convert OpenCV-style hand coords to the convention expected here:
        # flip x/y but keep z so geometry stays in front of the camera.
        verts[:, 0] *= -1.0
        verts[:, 1] *= -1.0
        verts = verts.unsqueeze(0)

        faces = torch.as_tensor(faces_np, dtype=torch.int64, device=device).unsqueeze(0)
        color = torch.tensor(mesh_base_color, dtype=torch.float32, device=device)
        vert_colors = color[None, None, :].expand(1, verts.shape[1], 3)
        textures = TexturesVertex(verts_features=vert_colors)
        mesh_list.append(Meshes(verts=verts, faces=faces, textures=textures))

    scene_mesh = join_meshes_as_scene(mesh_list)

    focal = float(focal_length)
    cameras = PerspectiveCameras(
        focal_length=((focal, focal),),
        principal_point=((width / 2.0, height / 2.0),),
        image_size=((height, width),),
        in_ndc=False,
        device=device,
    )
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    blend_params = BlendParams(background_color=scene_bg_color)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=blend_params,
        ),
    )

    rendered = renderer(scene_mesh)

    # Slightly boost alpha so overlays are easier to see.
    alpha_gain = 1.35
    rendered = rendered.clone()
    rendered[..., 3] = torch.clamp(rendered[..., 3] * alpha_gain, 0.0, 1.0)

    alpha = rendered[0, ..., 3]
    alpha_min = float(alpha.min().item())
    alpha_max = float(alpha.max().item())
    alpha_mean = float(alpha.mean().item())
    alpha_fg_pct = float((alpha > 1e-4).float().mean().item() * 100.0)
    if alpha_fg_pct < 0.01:
        print(
            "[render_rgba_multiple] warning: extremely low alpha coverage "
            f"(min={alpha_min:.6f}, max={alpha_max:.6f}, mean={alpha_mean:.6f}, fg%={alpha_fg_pct:.6f})"
        )

    return rendered[0, ..., :4].detach().cpu().numpy()


def overlay_rgba_on_bgr(bgr, rgba):
    """Alpha-compose an RGBA foreground onto a BGR background."""
    if bgr is None or rgba is None:
        raise ValueError("Both bgr and rgba inputs are required.")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected bgr shape (H,W,3), got {bgr.shape}")
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Expected rgba shape (H,W,4), got {rgba.shape}")

    out_dtype = bgr.dtype

    bgr_f = bgr.astype(np.float32)
    if np.issubdtype(out_dtype, np.integer):
        bgr_f = bgr_f / 255.0
    else:
        bgr_f = np.clip(bgr_f, 0.0, 1.0)

    rgba_f = rgba.astype(np.float32)
    if rgba_f.max() > 1.0:
        rgba_f = rgba_f / 255.0
    rgba_f = np.clip(rgba_f, 0.0, 1.0)

    if rgba_f.shape[:2] != bgr_f.shape[:2]:
        rgba_f = cv2.resize(rgba_f, (bgr_f.shape[1], bgr_f.shape[0]), interpolation=cv2.INTER_LINEAR)

    bg_rgb = bgr_f[:, :, ::-1]
    fg_rgb = rgba_f[:, :, :3]
    alpha = rgba_f[:, :, 3:4]

    out_rgb = fg_rgb * alpha + bg_rgb * (1.0 - alpha)
    out_bgr = np.clip(out_rgb[:, :, ::-1], 0.0, 1.0)

    if np.issubdtype(out_dtype, np.integer):
        return (out_bgr * 255.0).astype(out_dtype)
    return out_bgr.astype(out_dtype)
