import numpy as np
import open3d as o3d

from src.loaders.BaseLoader import BaseLoader
from src.model.SegmentedPlane import SegmentedPlane
from src.model.SegmentedPointCloud import SegmentedPointCloud


class CarlaLoader(BaseLoader):
    PLANAR_IDS = [1, 2, 7, 8, 11, 12]

    def __init__(self, path):
        super().__init__(path)

    def read_pcd(self, frame_num) -> SegmentedPointCloud:
        with open(self.path, 'r') as input_file:
            data = input_file.read()
            points, labels = CarlaLoader.CarlaParser.parse_cloud(data)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            planes = self.__get_all_planes(labels)
            all_segmented_indices = []
            for plane in planes:
                all_segmented_indices.append(plane.pcd_indices)

            return SegmentedPointCloud(
                pcd=pcd,
                planes=planes,
                unsegmented_cloud_indices=np.setdiff1d(
                    np.arange(points.shape[0]),
                    np.concatenate(all_segmented_indices)
                )
            )

    def get_frame_count(self) -> int:
        return 1

    def __get_all_planes(self, loaded_labels) -> list:
        unique_ids = np.unique(loaded_labels)
        planes = []
        for plane_id in unique_ids:
            if plane_id not in CarlaLoader.PLANAR_IDS:
                continue
            plane_indices = np.where(loaded_labels == plane_id)[0]
            planes.append(SegmentedPlane(
                pcd_indices=plane_indices,
                track_id=SegmentedPlane.NO_TRACK
            ))

        return planes

    class CarlaParser:
        @staticmethod
        def parse_cloud(cloud_string) -> (np.array, np.array):
            transform_separator_position = cloud_string.find("[") - 1
            end_data_position = cloud_string.rfind("]")
            data_string = cloud_string[transform_separator_position + 2:end_data_position]  # skip global []
            coords = []
            labels = []
            while len(data_string) > 0:
                start_of_point_data = data_string.find("[")
                # if start_of_point_data == -1:
                #     break
                start_of_point_coord = start_of_point_data + 1
                end_of_point_coord = data_string.find("]")
                end_of_point_data = data_string[end_of_point_coord + 1:].find("]") + (end_of_point_coord + 1)
                coord_string = data_string[start_of_point_coord + 1:end_of_point_coord]
                label_string = data_string[end_of_point_coord + 2:end_of_point_data]

                coords.append(np.fromstring(coord_string, dtype=float, sep=','))
                labels.append(int(label_string))

                data_string = data_string[end_of_point_data + 1:]

            return np.asarray(coords), np.asarray(labels)

