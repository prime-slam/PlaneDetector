from lxml import objectify

from src.utils.colors import get_random_color


def parse_points(points_str):
    points = []
    for point_str in points_str.split(';'):
        coords_str = point_str.split(',')
        coords = [float(coords_str[0]), float(coords_str[1])]
        points.append(coords)

    return points


class CVATAnnotation:
    def __init__(self, path: str, start_frame_num: int):
        root = objectify.parse(path).getroot()
        tracks = [child for child in root.iterchildren()][2:]

        self.tracks = []
        self.start_frame_num = start_frame_num
        self.min_frame_id = None
        self.max_frame_id = None

        for track in tracks:
            frames = track.getchildren()
            planes_track = self.Track()
            for frame in frames:
                points = parse_points(frame.attrib['points'])
                frame_id = int(frame.attrib['frame'])

                if self.min_frame_id is None or frame_id < self.min_frame_id:
                    self.min_frame_id = frame_id
                if self.max_frame_id is None or frame_id > self.max_frame_id:
                    self.max_frame_id = frame_id

                is_outside = int(frame.attrib['outside'])
                if is_outside == 1:
                    continue

                plane = self.Plane(points)
                planes_track.append(plane, frame_id)

            self.tracks.append(planes_track)

    def get_max_frame_id(self):
        return self.max_frame_id + self.start_frame_num

    def get_min_frame_id(self):
        return self.min_frame_id + self.start_frame_num

    def get_plane_by_track_and_frame(self, track_id, frame_id):
        frame_id -= self.start_frame_num
        return self.tracks[track_id].planes[frame_id]

    def get_all_planes_for_frame(self, frame_id):
        frame_id -= self.start_frame_num
        return [track.planes[frame_id] for track in self.tracks if frame_id in track.planes]

    class Track:
        def __init__(self):
            self.planes = {}
            self.color = get_random_color()

        def append(self, plane, frame_id):
            self.planes[frame_id] = plane
            plane.color = self.color

    class Plane:
        def __init__(self, points):
            self.points = points
            self.color = (0, 0, 0)
