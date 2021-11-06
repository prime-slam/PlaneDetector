import numpy as np
import open3d as o3d
import pytest

from src.utils.metrics import pcd_delta


def test_pcd_delta():
    left_points = np.array([
        [1., 2., 3.],
        [0.5, 0.6, 0.3],
        [0., 0., 0.]
    ])
    right_points = np.array([
        [6.6, 7.8, 0.],
        [0., 0., 0.]
    ])
    res_points_gt = np.array([
        [1., 2., 3.],
        [0.5, 0.6, 0.3]
    ])
    left_pcd = o3d.geometry.PointCloud()
    left_pcd.points = o3d.utility.Vector3dVector(left_points)
    right_pcd = o3d.geometry.PointCloud()
    right_pcd.points = o3d.utility.Vector3dVector(right_points)

    res = pcd_delta(left_pcd, right_pcd)
    res_points = np.array(res.points)
    np.testing.assert_almost_equal(res_points, res_points_gt)
