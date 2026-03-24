from typing import Optional
import torch
from torch.nn import functional as F

def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Works by first converting it to a quaternion.
    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.
    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    x = x.reshape(-1,2,3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

def perspective_projection(points: torch.Tensor,
                           translation: torch.Tensor,
                           focal_length: torch.Tensor,
                           camera_center: Optional[torch.Tensor] = None,
                           rotation: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the perspective projection of a set of 3D points.
    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        focal_length (torch.Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 2) containing the projection of the input points.
    """
    batch_size = points.shape[0]
    if rotation is None:
        rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    if camera_center is None:
        camera_center = torch.zeros(batch_size, 2, device=points.device, dtype=points.dtype)
    # Populate intrinsic camera matrix K.
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def cam_crop_to_full(
    cam_bbox: torch.Tensor,
    box_center: torch.Tensor,
    box_size: torch.Tensor,
    img_size: torch.Tensor,
    focal_length: torch.Tensor,
) -> torch.Tensor:
    """
    Convert weak-perspective crop camera parameters to full-image translation.
    Args:
        cam_bbox (torch.Tensor): Tensor of shape (B, 3) with weak-perspective
            camera parameters [scale, tx, ty].
        box_center (torch.Tensor): Tensor of shape (B, 2) with bbox center.
        box_size (torch.Tensor): Tensor of shape (B,) with square bbox size.
        img_size (torch.Tensor): Tensor of shape (B, 2) with [width, height].
        focal_length (torch.Tensor): Tensor of shape (B,) or (B, 2).
    Returns:
        torch.Tensor: Tensor of shape (B, 3) with full-image camera translation.
    """
    if focal_length.ndim == 2:
        focal_length = focal_length[:, 0]
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy = box_center[:, 0], box_center[:, 1]
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = box_size * cam_bbox[:, 0] + 1e-9
    tz = 2.0 * focal_length / bs
    tx = (2.0 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2.0 * (cy - h_2) / bs) + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


def compute_full_image_camera_translation(
    pred_cam: torch.Tensor,
    box_center: torch.Tensor,
    box_size: torch.Tensor,
    img_size: torch.Tensor,
    focal_length_base: float,
    model_image_size: int,
    right: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert WiLoR camera output to full-image translation using inference-time math.
    Returns the translation and the scaled full-image focal length per sample.
    """
    pred_cam_full = pred_cam.clone()
    if right is not None:
        multiplier = 2.0 * right.float() - 1.0
        pred_cam_full[:, 1] = multiplier * pred_cam_full[:, 1]

    img_size_float = img_size.float()
    scaled_focal_length = (
        float(focal_length_base) / float(model_image_size) * img_size_float.max(dim=1).values
    )
    pred_cam_t_full = cam_crop_to_full(
        pred_cam_full,
        box_center.float(),
        box_size.float(),
        img_size_float,
        scaled_focal_length,
    )
    return pred_cam_t_full, scaled_focal_length
