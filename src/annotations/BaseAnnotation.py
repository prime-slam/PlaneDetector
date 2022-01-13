from abc import ABC


class BaseAnnotation(ABC):
    def __init__(self, path):
        self.path = path

    def get_max_frame_id(self):
        pass

    def get_min_frame_id(self):
        pass

    def get_plane_by_track_and_frame(self, track_id, frame_id):
        pass

    def get_all_planes_for_frame(self, frame_id):
        pass
