import torch
import numpy as np
import trimesh
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import MeshRenderer,MeshRasterizer,SoftPhongShader,RasterizationSettings,PerspectiveCameras,PointLights,TexturesVertex,BlendParams

# from pytorch3d.renderer import (
#     MeshRenderer,
#     MeshRasterizer,
#     SoftPhongShader,
#     RasterizationSettings,
#     PerspectiveCameras,
#     PointLights,
#     TexturesVertex
# )
def render_rgba_multiple(
    vertices,
    faces_in, 
    cam_t,
    rot_axis=[1,0,0],
    rot_angle=0,
    mesh_base_color=(1.0, 1.0, 0.9),
    scene_bg_color=(0,0,0),
    render_res=[256, 256],
    focal_length=None,
    is_right=None,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_right is None:
        is_right = [1 for _ in range(len(vertices))]

    if focal_length is None:
        raise ValueError("focal_length must be provided for render_rgba_multiple")

    render_res = (int(render_res[0]), int(render_res[1]))  # (W, H)
    width, height = render_res

    if torch.is_tensor(faces_in):
        faces_right = faces_in.detach().cpu().numpy().copy()
    else:
        faces_right = np.asarray(faces_in).copy()
    faces_left = faces_right[:, [0, 2, 1]]

    mesh_list = []
    for vvv, ttt, sss in zip(vertices, cam_t, is_right):
        faces_used = faces_right if int(sss) == 1 else faces_left

        mesh_trimesh = vertices_to_trimesh(
            vvv, faces_used, ttt.copy(), mesh_base_color, rot_axis, rot_angle
        )

        verts = torch.tensor(mesh_trimesh.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(mesh_trimesh.faces, dtype=torch.int64, device=device)

        verts = verts.unsqueeze(0)
        faces = faces.unsqueeze(0)

        vertex_color = torch.tensor(mesh_base_color, device=device).float()
        vertex_color = vertex_color[None, None, :].expand(1, verts.shape[1], 3)

        textures = TexturesVertex(verts_features=vertex_color)

        mesh = Meshes(verts=verts, faces=faces, textures=textures)
        mesh_list.append(mesh)

    scene_mesh = join_meshes_as_scene(mesh_list)

    cameras = PerspectiveCameras(
        focal_length=((float(focal_length), float(focal_length)),),
        principal_point=((width / 2.0, height / 2.0),),
        image_size=((height, width),),
        in_ndc=False,
        device=device
    )

    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    blend_params = BlendParams(background_color=scene_bg_color)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=blend_params
        )
    )

    images = renderer(scene_mesh)

    return images[0, ..., :4].detach().cpu().numpy()


def vertices_to_trimesh(
    vertices,
    faces,
    camera_translation,
    mesh_base_color=(1.0, 1.0, 0.9),
    rot_axis=(1, 0, 0),
    rot_angle=0,
):

    vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
    verts = vertices.copy() + camera_translation

    mesh = trimesh.Trimesh(
        verts,
        faces.copy(),
        # faces.detach().cpu().numpy().copy(),
        vertex_colors=vertex_colors
    )

    if rot_angle != 0:
        rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), rot_axis
        )
        mesh.apply_transform(rot)

    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0]
    )
    mesh.apply_transform(rot)

    return mesh


def vertices_to_pytorch3d_mesh(
    vertices,
    faces,
    faces_left,
    camera_translation,
    mesh_base_color=(1.0, 1.0, 0.9),
    rot_axis=(1, 0, 0),
    rot_angle=0,
    is_right=1,
    device="cuda"
):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    verts = vertices.copy() + camera_translation

    faces_used = faces if is_right else faces_left

    verts = torch.tensor(verts, dtype=torch.float32, device=device)
    faces = torch.tensor(faces_used.copy(), dtype=torch.int64, device=device)

    # Apply arbitrary rotation
    if rot_angle != 0:
        rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), rot_axis
        )[:3, :3]
        rot = torch.tensor(rot, dtype=torch.float32, device=device)
        verts = verts @ rot.T

    # Apply fixed 180° flip
    rot180 = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0]
    )[:3, :3]
    rot180 = torch.tensor(rot180, dtype=torch.float32, device=device)
    verts = verts @ rot180.T

    verts = verts.unsqueeze(0)
    faces = faces.unsqueeze(0)

    color = torch.tensor(mesh_base_color, device=device).float()
    color = color[None, None, :].expand(1, verts.shape[1], 3)

    textures = TexturesVertex(verts_features=color)

    return Meshes(verts=verts, faces=faces, textures=textures)

def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    # Convert cam_bbox to full image
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam