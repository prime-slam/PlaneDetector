import open3d as o3d


class CameraIntrinsics:
    def __init__(self, width, height, fx, fy, cx, cy, factor):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.factor = factor
        if fx is not None and fy is not None and cx is not None and cy is not None:
            self.open3dIntrinsics = o3d.camera.PinholeCameraIntrinsic(
                width=width,
                height=height,
                fx=fx,  # X-axis focal length
                fy=fy,  # Y-axis focal length
                cx=cx,  # X-axis principle point
                cy=cy,  # Y-axis principle point
            )
