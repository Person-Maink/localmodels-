import os
import cv2
import imageio
import numpy as np
import torch
import subprocess

from body_model import run_mano
from geometry import camera as cam_util
from geometry.mesh import make_batch_mesh, save_mesh_scenes
from geometry.plane import parse_floor_plane, get_plane_transform

from util.tensor import detach_all, to_torch, move_to

from .fig_specs import get_seq_figure_skip, get_seq_static_lookat_points
from .tools import mesh_to_geometry, smooth_results


def prep_result_vis(res, vis_mask, track_ids, body_model, temporal_smooth):
    """
    :param res (dict) with (B, T, *) tensor elements, B tracks and T frames
    :param vis_mask (B, T) with visibility of each track in each frame
    :param track_ids (B,) index of each track
    """
    print("RESULT FIELDS", res.keys())
    res = detach_all(res)
    # print(res.keys(), res["is_right"])
    with torch.no_grad():
        if temporal_smooth:
            print('running temporal smooth')
            res["root_orient"], res["pose_body"], res['betas'], res["trans"] = smooth_results(res["root_orient"], res["pose_body"], res['betas'], res["is_right"], res["trans"])

        world_smpl = run_mano(
            body_model,
            res["trans"],
            res["root_orient"],
            res["pose_body"],
            res["is_right"],
            res.get("betas", None),
        )

    T_w2c = None
    floor_plane = None
    if "cam_R" in res and "cam_t" in res:
        T_w2c = cam_util.make_4x4_pose(res["cam_R"][0], res["cam_t"][0])
    if "floor_plane" in res:
        floor_plane = res["floor_plane"][0]
    return build_scene_dict(
        world_smpl,
        vis_mask,
        track_ids,
        T_w2c=T_w2c,
        floor_plane=floor_plane,
    )


def build_scene_dict(
    world_smpl, vis_mask, track_ids, T_w2c=None, floor_plane=None, **kwargs
):
    scene_dict = {}

    # first get the geometry of the people
    # lists of length T with (B, V, 3), (F, 3), (B, 3)
    scene_dict["geometry"] = mesh_to_geometry(
        world_smpl["vertices"], world_smpl["joints"], world_smpl["l_faces"], world_smpl["r_faces"], world_smpl["is_right"], vis_mask, track_ids
    )

    if T_w2c is None:
        T_w2c = torch.eye(4)[None]

    T_c2w = torch.linalg.inv(T_w2c)
    # rotate the camera slightly down and translate back and up
    T = cam_util.make_4x4_pose(
        cam_util.rotx(-np.pi / 10), torch.tensor([0, -1, -2])
    ).to(T_c2w.device)

    scene_dict["cameras"] = {
        "src_cam": T_c2w,
        # "front": torch.einsum("ij,...jk->...ik", T, T_c2w),
    }

    # Create ground plane based on hand mesh height
    verts = scene_dict["geometry"][0]  # Get vertices from geometry
    if len(verts) > 0:
        # Find minimum height across all frames and vertices
        min_height = float('inf')
        for frame_verts in verts:
            if len(frame_verts) > 0:
                frame_min = frame_verts[..., 1].min().item()  # y-coordinate is height
                min_height = min(min_height, frame_min)
        
        # Set ground plane further below minimum height
        ground_offset = -0.5  # 20cm below minimum height
        ground_height = min_height - ground_offset
        
        # Create ground plane transform
        R = torch.eye(3)  # Identity rotation (flat ground)
        t = torch.tensor([0.0, ground_height, 0.0])  # Translate to ground height
        scene_dict["ground"] = cam_util.make_4x4_pose(R, t)

        # Save ground mesh for Blender debugging
        import trimesh
        from vis.viewer import make_checkerboard
        ground_mesh = make_checkerboard(color0=[0.9, 0.95, 1.0], color1=[0.7, 0.8, 0.85], up="y", alpha=1.0)
        ground_mesh.apply_translation([0.0, ground_height, 0.0])
        ground_mesh.export("ground_debug.obj")

    # if floor_plane is not None:
    #     # compute the ground transform
    #     # use the first appearance of a track as the reference point
    #     tid, sid = torch.where(vis_mask > 0)
    #     idx = tid[torch.argmin(sid)]
    #     root = world_smpl["joints"][idx, 0, 0].detach().cpu()
    #     floor = parse_floor_plane(floor_plane.detach().cpu())
    #     R, t = get_plane_transform(torch.tensor([0.0, 1.0, 0.0]), floor, root)
    #     scene_dict["ground"] = cam_util.make_4x4_pose(R, t)

    return scene_dict


# def render_scene_dict(renderer, scene_dict, out_name, fps=30, **kwargs):
#     # lists of T (B, V, 3), (B, 3), (F, 3)
#     verts, colors, faces, bounds = scene_dict["geometry"]
#     print("NUM VERTS", len(verts))

#     # add a top view
#     scene_dict["cameras"]["above"] = cam_util.make_4x4_pose(
#         torch.eye(3), torch.tensor([0, 0, -10])
#     )[None]

#     for cam_name, cam_poses in scene_dict["cameras"].items():
#         print("rendering scene for", cam_name)
#         # cam_poses are (T, 4, 4)
#         render_bg = cam_name == "src_cam"
#         ground_pose = scene_dict.get("ground", None)
#         print(cam_name, ground_pose)
#         frames = renderer.render_video(
#             cam_poses[None], verts, faces, colors, render_bg, ground_pose=ground_pose
#         )
#         os.makedirs(f"{out_name}_{cam_name}/", exist_ok=True)
#         for idx, i in enumerate(frames):
#             imageio.imwrite(f"{out_name}_{cam_name}/" + f'{str(idx).zfill(6)}.jpg', i)
#         imageio.mimwrite(f"{out_name}_{cam_name}.mp4", frames, fps=fps)


def animate_scene(
    vis,
    scene,
    out_name,
    seq_name=None,
    accumulate=False,
    render_views=["src_cam", "front", "above", "side"],
    render_bg=True,
    render_cam=True,
    render_ground=True,
    debug=False,
    **kwargs,
):
    if len(render_views) < 1:
        return

    scene = build_pyrender_scene(
        vis,
        scene,
        seq_name,
        render_views=render_views,
        render_cam=render_cam,
        accumulate=accumulate,
        debug=debug,
    )

    print("RENDERING VIEWS", scene["cameras"].keys())
    render_ground = render_ground and "ground" in scene
    save_paths = []
    for cam_name, cam_poses in scene["cameras"].items():
        is_src = cam_name == "src_cam"
        show_bg = is_src and render_bg
        show_ground = render_ground and not is_src
        show_cam = render_cam and not is_src
        vis_name = f"{out_name}_{cam_name}"
        print(f"{cam_name} has {len(cam_poses)} poses")
        skip = 10 if debug else 1
        vis.set_camera_seq(cam_poses[::skip])
        save_path = vis.animate(
            vis_name,
            render_bg=show_bg,
            render_ground=show_ground,
            render_cam=show_cam,
            **kwargs,
        )
        save_paths.append(save_path)

    return save_paths


def build_pyrender_scene(
    vis,
    scene,
    seq_name,
    render_views=["src_cam", "front", "above", "side"],
    render_cam=True,
    accumulate=False,
    debug=False,
):
    """
    :param vis (viewer object)
    :param scene (scene dict with geometry, cameras, etc)
    :param accumulate (optional bool, default False) whether to render entire trajectory together
    :param render_views (list str) camera views to render
    """
    if len(render_views) < 1:
        return

    assert all(view in ["src_cam", "front", "above", "side"] for view in render_views)

    scene = move_to(detach_all(scene), "cpu")
    src_cams = scene["cameras"]["src_cam"]
    verts, _, colors, l_faces, r_faces, is_right, bounds = scene["geometry"]
    T = len(verts)
    print(f"{T} mesh frames for {seq_name}, {len(verts)}")

    # set camera views
    if not "cameras" in scene:
        scene["cameras"] = {}

    # remove default views from source camera perspective if desired
    if "src_cam" not in render_views:
        scene["cameras"].pop("src_cam", None)
    if "front" not in render_views:
        scene["cameras"].pop("front", None)

    # add static viewpoints if desired
    top_pose, side_pose, _skip = get_static_views(seq_name, bounds)
    if "above" in render_views:
        scene["cameras"]["above"] = top_pose[None]
    if "side" in render_views:
        scene["cameras"]["side"] = side_pose[None]
    if "front" in render_views:
        # Use a static front camera: place it in front of the hand mesh, looking at the center
        # Place the camera at a fixed distance along the -z axis from the center
        if bounds is not None:
            bb_min, bb_max, center = bounds
            length = torch.abs(bb_max - bb_min).max()
            front_offset = 1.0  # 1 closer, 2 farther
            front_source = center + torch.tensor([0.0, -0.5, -front_offset])
            front_target = center
            up = torch.tensor([0.0, 1, 0.0])
            front_pose = cam_util.lookat_matrix(front_source, front_target, up)
            scene["cameras"]["front"] = front_pose[None]
        else:
            # fallback to previous behavior if bounds not available
            pass

    # accumulate meshes if possible (can only accumulate for static camera)
    moving_cam = "src_cam" in render_views or "front" in render_views
    accumulate = accumulate and not moving_cam
    skip = _skip if accumulate else 1

    vis.clear_meshes()

    if "ground" in scene:
        vis.set_ground(scene["ground"])

    if debug:
        skip = 10
    times = list(range(0, T, skip))
    # print(times, verts[0].shape)
    # raise ValueError
    for t in times:
        if len(is_right[t]) > 1:
            assert (is_right[t].cpu().numpy().tolist() == [0,1])
            l_meshes = make_batch_mesh(verts[t][0][None], l_faces[t], colors[t][0][None])
            r_meshes = make_batch_mesh(verts[t][1][None], r_faces[t], colors[t][1][None])
            assert len(l_meshes) == 1
            assert len(r_meshes) == 1
            meshes = [l_meshes[0], r_meshes[0]]
        else:
            assert len(is_right[t]) == 1
            if is_right[t] == 0:
                meshes = make_batch_mesh(verts[t][0][None], l_faces[t], colors[t][0][None])
            elif is_right[t] == 1:
                meshes = make_batch_mesh(verts[t][0][None], r_faces[t], colors[t][0][None])

        if accumulate:
            vis.add_static_meshes(meshes)
        else:
            vis.add_mesh_frame(meshes, debug=debug)

    # add camera markers
    # if render_cam:
    #     if accumulate:
    #         vis.add_camera_markers_static(src_cams[::skip])
    #     else:
    #         vis.add_camera_markers(src_cams[::skip])

    return scene


def get_static_views(seq_name=None, bounds=None):
    print("STATIC VIEWS FOR SEQ NAME", seq_name)
    up = torch.tensor([0.0, 1.0, 0.0])

    skip = get_seq_figure_skip(seq_name)
    top_vp, side_vp = get_seq_static_lookat_points(seq_name, bounds)
    top_source, top_target = top_vp
    side_source, side_target = side_vp
    top_pose = cam_util.lookat_matrix(top_source, top_target, up)
    side_pose = cam_util.lookat_matrix(side_source, side_target, up)
    return top_pose, side_pose, skip


def make_video_grid_2x2(out_path, vid_paths, overwrite=False):
    if os.path.isfile(out_path) and not overwrite:
        print(f"{out_path} already exists, skipping.")
        return

    if any(not os.path.isfile(v) for v in vid_paths):
        print("not all inputs exist!", vid_paths)
        return

    # resize each input by half and then tile
    # so the output video is the same resolution
    # v1, v2, v3, v4 = vid_paths
    # cmd = (
    #     f"ffmpeg -i {v1} -i {v2} -i {v3} -i {v4} "
    #     f"-filter_complex '[0:v]scale=iw/2:ih/2[v0];"
    #     f"[1:v]scale=iw/2:ih/2[v1];"
    #     f"[2:v]scale=iw/2:ih/2[v2];"
    #     f"[3:v]scale=iw/2:ih/2[v3];"
    #     f"[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]' "
    #     f"-map '[v]' {out_path} -y"
    # )

    # print(cmd)
    # subprocess.call(cmd, shell=True, stdin=subprocess.PIPE)

    # 读取四个视频
    videos = [cv2.VideoCapture(v) for v in vid_paths]

    # Check if videos are opened successfully
    if any(not video.isOpened() for video in videos):
        print("Error opening input videos.")
        return

    # Get video properties (assuming all videos have the same properties)
    width = int(videos[0].get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    height = int(videos[0].get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
    fps = videos[0].get(cv2.CAP_PROP_FPS)

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose a different codec if needed
    out = cv2.VideoWriter(out_path, fourcc, fps, (width * 2, height * 2))

    # Read and resize frames, then stack them
    while True:
        frames = [video.read()[1] for video in videos]
        if any(frame is None for frame in frames):
            break
        else:
            frames = [cv2.resize(frame, (width, height)) for frame in frames]
        
        stacked_frame = np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:])])
        out.write(stacked_frame)

    # Release VideoCapture and VideoWriter objects
    for video in videos:
        video.release()
    out.release()

    print(f"Video grid created: {out_path}")