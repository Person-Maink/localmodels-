import torch
import numpy as np
import trimesh
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import MeshRenderer,MeshRasterizer,SoftPhongShader,RasterizationSettings,PerspectiveCameras,PointLights,TexturesVertex

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
    faces, 
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

    mesh_list = []

    for vvv, ttt, sss in zip(vertices, cam_t, is_right):

        mesh_trimesh = vertices_to_trimesh(
            vvv, faces, ttt.copy(), mesh_base_color, rot_axis, rot_angle
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

    scene_mesh = join_meshes_as_batch(mesh_list)

    # focal_length = focal_length if focal_length is not None else focal_length
    if not focal_length: 
        print("FOCAL_LENGTH IS NONE")

    cameras = PerspectiveCameras(
        focal_length=((focal_length, focal_length),),
        principal_point=((render_res[0]/2, render_res[1]/2),),
        image_size=((render_res[1], render_res[0]),),
        device=device
    )

    raster_settings = RasterizationSettings(
        image_size=render_res,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
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



# def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9), 
#                         rot_axis=[1,0,0], rot_angle=0, is_right=1):
#     # material = pyrender.MetallicRoughnessMaterial(
#     #     metallicFactor=0.0,
#     #     alphaMode='OPAQUE',
#     #     baseColorFactor=(*mesh_base_color, 1.0))
#     vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
#     if is_right:
#         mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
#     else:
#         mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces_left.copy(), vertex_colors=vertex_colors)
#     # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
    
#     rot = trimesh.transformations.rotation_matrix(
#             np.radians(rot_angle), rot_axis)
#     mesh.apply_transform(rot)

#     rot = trimesh.transformations.rotation_matrix(
#         np.radians(180), [1, 0, 0])
#     mesh.apply_transform(rot)
#     return mesh


# def render_rgba_multiple(
#         self,
#         vertices: List[np.array],
#         cam_t: List[np.array],
#         rot_axis=[1,0,0],
#         rot_angle=0,
#         mesh_base_color=(1.0, 1.0, 0.9),
#         scene_bg_color=(0,0,0),
#         render_res=[256, 256],
#         focal_length=None,
#         is_right=None,
#     ):

#     renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
#                                             viewport_height=render_res[1],
#                                             point_size=1.0)
#     # material = pyrender.MetallicRoughnessMaterial(
#     #     metallicFactor=0.0,
#     #     alphaMode='OPAQUE',
#     #     baseColorFactor=(*mesh_base_color, 1.0))

#     if is_right is None:
#         is_right = [1 for _ in range(len(vertices))]

#     mesh_list = [pyrender.Mesh.from_trimesh(self.vertices_to_trimesh(vvv, ttt.copy(), mesh_base_color, rot_axis, rot_angle, is_right=sss)) for vvv,ttt,sss in zip(vertices, cam_t, is_right)]

#     scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
#                             ambient_light=(0.3, 0.3, 0.3))
#     for i,mesh in enumerate(mesh_list):
#         scene.add(mesh, f'mesh_{i}')

#     camera_pose = np.eye(4)
#     # camera_pose[:3, 3] = camera_translation
#     camera_center = [render_res[0] / 2., render_res[1] / 2.]
#     focal_length = focal_length if focal_length is not None else self.focal_length
#     camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
#                                         cx=camera_center[0], cy=camera_center[1], zfar=1e12)

#     # Create camera node and add it to pyRender scene
#     camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
#     scene.add_node(camera_node)
#     self.add_point_lighting(scene, camera_node)
#     self.add_lighting(scene, camera_node)

#     light_nodes = create_raymond_lights()
#     for node in light_nodes:
#         scene.add_node(node)

#     color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
#     color = color.astype(np.float32) / 255.0
#     renderer.delete()

#     return color