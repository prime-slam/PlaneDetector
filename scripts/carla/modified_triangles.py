import os
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy

from src.utils.colors import get_random_normalized_color

def filter_small_triangles(mesh: o3d.geometry.TriangleMesh, min_area: float, min_ratio: float):
    #  filter small triangles
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    areas = np.ones(triangles.shape[:-1], dtype=float)
    mask = np.zeros(triangles.shape[:-1], dtype=bool)
    for triangle_index, triangle in enumerate(triangles):
        p1, p2, p3 = vertices[triangle]

        # These two vectors are in the plane
        v1 = p3 - p1
        v2 = p2 - p1
        v3 = p2 - p3
        lens = np.sort(np.array([np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)]))
        # the cross product is a vector normal to the plane
        cp = np.cross(v1, v2)
        areas[triangle_index] = np.linalg.norm(cp) / 2
        if not(np.linalg.norm(cp) / 2 > min_area and lens[1] / lens[0]> min_ratio):
            # This evaluates a * x3 + b * y3 + c * z3 which equals d
            mask[triangle_index] = True
    mesh.remove_triangles_by_mask(mask)
    # o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, mesh_show_back_face=True)

    return mesh


def process_mesh(mesh_file_path: str, output_path: str, min_area: float, min_ratio: float):
    plane_mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    o3d.visualization.draw_geometries([plane_mesh], mesh_show_wireframe=True, mesh_show_back_face=True)

    # plane_mesh = filter_small_triangles(plane_mesh, min_area, min_ratio)

    #  calculate planes
    plane_mesh.compute_triangle_normals(normalized=True)
    normals = np.asarray(plane_mesh.triangle_normals)
    
    triangles = np.asarray(plane_mesh.triangles)
    vertices = np.asarray(plane_mesh.vertices)

    planes = []
    for triangle_index, triangle in enumerate(triangles):
        d = np.dot(normals[triangle_index], vertices[triangle[2]])
        plane = np.concatenate([normals[triangle_index], np.asarray([d])])
        planes.append(plane)
    planes = np.asarray(planes)

    clustering = DBSCAN(eps=1, min_samples=1, n_jobs=20).fit(planes)
    label_to_meshes = defaultdict(list)
    unique_labels = np.unique(clustering.labels_)
    for plane_id in unique_labels:
        label_to_meshes[plane_id] = np.where(clustering.labels_ == plane_id)[0]
    

    #  construct meshes out of planes
    planes_meshes = []
    for label_triangles in label_to_meshes.values():
        # triangles_3 = []
        # vertices_3 = []
        mesh_triangles_for_plane = triangles[label_triangles]
        plane_mesh_vertices = vertices[mesh_triangles_for_plane.flatten()]
        triangles_count = len(label_triangles)
        plane_mesh_triangles = np.arange(triangles_count * 3).reshape((triangles_count, 3))
        # for i, triangle_id in enumerate(label_triangles):
        #     triangle = triangles[triangle_id]
        #     vertices_3.append(vertices[triangle[0]])
        #     vertices_3.append(vertices[triangle[1]])
        #     vertices_3.append(vertices[triangle[2]])
        #
        #     triangles_3.append(np.arange(i*3, i*3 + 3))
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.triangles = o3d.utility.Vector3iVector(plane_mesh_triangles)
        new_mesh.vertices = o3d.utility.Vector3dVector(plane_mesh_vertices)
        # new_mesh.triangles = o3d.utility.Vector3iVector(triangles_3)
        # new_mesh.vertices = o3d.utility.Vector3dVector(vertices_3)
        if new_mesh.get_surface_area() == 0:
            continue
        filtered_mesh = filter_small_triangles(deepcopy(new_mesh), min_area, min_ratio)
        #experiments show that "good" planes consist of minimum 2 triangles 
        if len(np.asarray(filtered_mesh.triangles)) < 2:
            continue
        planes_meshes.append(new_mesh)
    
    # sample and color mesh-planes
    plane_instance_pcds = []
    for plane_mesh in planes_meshes:
        pcd = plane_mesh.sample_points_poisson_disk(
            int(np.sqrt(plane_mesh.get_surface_area()) / 2)
        )
        pcd.paint_uniform_color([0, 0, 0])
        points = np.asarray(pcd.points)

        # As plane can have spaces because was selected by normals -> need to take local parts divided
        clustering = DBSCAN(eps=100, min_samples=200, n_jobs=20).fit(points)
        for plane_id in np.unique(clustering.labels_):
            plane_instance_pcd = o3d.geometry.PointCloud()
            plane_instance_points = points[np.where(clustering.labels_ == plane_id)[0]]
            plane_instance_pcd.points = o3d.utility.Vector3dVector(plane_instance_points)
            plane_instance_pcds.append(plane_instance_pcd)
            plane_instance_pcd.paint_uniform_color(get_random_normalized_color())

            # o3d.visualization.draw_geometries([plane_instance_pcd])

    # o3d.visualization.draw_geometries(plane_instance_pcds)

    mesh_name = os.path.split(mesh_file_path)[-1][:-4]
    output_filename = "{}.pcd".format(mesh_name)
    labels_filename = "{}.npy".format(mesh_name)
    o3d.io.write_point_cloud(
        os.path.join(output_path, output_filename),
        sum(plane_instance_pcds, start=o3d.geometry.PointCloud())
    )
    labels = np.concatenate(
        [np.repeat(index + 1, np.asarray(pcd.points).shape[0]) for index, pcd in enumerate(plane_instance_pcds)]
    )
    np.save(os.path.join(output_path, labels_filename), labels)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--min_area',
        type=float,
        default=0,
        help='the minimal area of triangles that will not be filtered'
    )
    argparser.add_argument(
        '--min_ratio',
        type=float,
        default=0,
        help='the minimal ratio of triangle sides that will not be filtered'
    )
    argparser.add_argument(
        'mesh',
        help='path to mesh'
    )
    argparser.add_argument(
        'output_path',
        help='path to where to save pcd'
    )
    args = argparser.parse_args()

    mesh_filenames = os.listdir(args.mesh)
    for index, mesh_filename in enumerate(mesh_filenames):
        mesh_path = os.path.join(args.mesh, mesh_filename)
        process_mesh(mesh_path, args.output_path, args.min_area, args.min_ratio)
        print("{0} is ready! ({1}/{2})".format(mesh_filename[:-4], index + 1, len(mesh_filenames)))
