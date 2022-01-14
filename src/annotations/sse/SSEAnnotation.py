import numpy as np

from src.annotations.sse.FIC import FIC
from src.annotations.sse.LZW import LZW


class SSEAnnotation:
    def __init__(self, path):
        self.path = path

    def __load_labels(self) -> np.array:
        with open(self.path, "rb") as label_file:
            data = bytearray(label_file.read())
            labels_string = LZW.decompress(FIC.decompress(data))
            labels_string = labels_string[1:-1]  # skip []
            return np.asarray(list(map(lambda x: int(x), labels_string.split(","))))

    def get_all_planes(self) -> list:
        loaded_labels = self.__load_labels()
        unique_ids = np.unique(loaded_labels)
        planes = []
        for plane_id in unique_ids:
            # 0 label is for unsegmented areas
            if plane_id == 0:
                continue
            plane_indices = np.where(loaded_labels == plane_id)[0]
            planes.append(SSEAnnotation.Plane(plane_indices))

        return planes

    class Plane:
        def __init__(self, indices):
            self.indices = indices
