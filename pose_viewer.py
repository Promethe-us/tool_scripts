import open3d as o3d
import torch
import numpy as np

def create_camera_actor(is_gt=False, scale=1.0):
    cam_points = scale * np.array([
        [0,   0,   0],
        [-1,  -1, 1.5],
        [1,  -1, 1.5],
        [1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5]])

    cam_lines = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [1, 3], [2, 4],
                          [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])
    points = []
    for cam_line in cam_lines:
        begin_points, end_points = cam_points[cam_line[0]
                                              ], cam_points[cam_line[1]]
        t_vals = np.linspace(0., 1., 100)
        begin_points, end_points
        point = begin_points[None, :] * \
            (1.-t_vals)[:, None] + end_points[None, :] * (t_vals)[:, None]
        points.append(point)
    points = np.concatenate(points)
    color = (0.0, 0.0, 0.0) if is_gt else (1.0, .0, .0)
    camera_actor = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(points))
    camera_actor.paint_uniform_color(color)

    return camera_actor

pts_pth = torch.load("D:/DESKTOP/0321/Vanilla3DGS/deep_blending_standard/playroom/pts/pts.pth")
xyz, rgb = pts_pth['xyz'].cpu().numpy(), pts_pth['rgb'].cpu().numpy()
pts_pcd = o3d.geometry.PointCloud()
pts_pcd.points = o3d.utility.Vector3dVector(xyz)
pts_pcd.colors = o3d.utility.Vector3dVector(rgb/255.0)

all_record = o3d.geometry.PointCloud()



def check_pcd_with_poses(pcd, poses, interp=1):
    '''
    pcd是ply
    poses是列表，每一项是4x4的变换矩阵
    interp是间隔
    '''
    all_record = o3d.geometry.PointCloud()
    for idx in range(0,len(poses),interp):
        pose = poses[idx]
        cam_actor = create_camera_actor()
        cam_actor.transform(pose)
        all_record += cam_actor
        
    o3d.visualization.draw_geometries([pcd, all_record])