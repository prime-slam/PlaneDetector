import os

import open3d as o3d

if __name__ == "__main__":
    path_dir = "debug"
    filenames = sorted(os.listdir(path_dir), key=lambda x: int(x.split(".")[0]))
    #
    # parts = {
    #     'part_-148_1_367_517_u45.pcd': (1, 52),
    #     'part_151_301_-82_67_u1.pcd': (53, 137),
    #     'part_151_301_367_517_u1.pcd': (138, 206),
    #     'part_151_301_67_217_u1.pcd': (207, 336),
    #     'part_1_151_-82_67_u24.pcd': (337, 409),
    #     'part_1_151_367_517_u3.pcd': (410, 481)
    # }

    # eye
    parts = {
        'part_-148_1_67_217_u74.pcd': (1, 72),
        'part_1_151_67_217_u73.pcd': (73, 178)
    }

    # res = {
    #     'part_-148_1_367_517_u45.pcd': [],
    #     'part_151_301_-82_67_u1.pcd': [],
    #     'part_151_301_367_517_u1.pcd': [],
    #     'part_151_301_67_217_u1.pcd': [],
    #     'part_1_151_-82_67_u24.pcd': [],
    #     'part_1_151_367_517_u3.pcd': []
    # }

    # eye
    res = {
        'part_-148_1_67_217_u74.pcd': [],
        'part_1_151_67_217_u73.pcd': []
    }

    for filename in filenames:
        label_id = int(filename.split(".")[0])
        for key, value in parts.items():
            min_val, max_val = value
            if min_val <= label_id <= max_val:
                res[key].append(label_id - min_val)

    # label VOID isn't used anywhere for this task
    label_names = [
        "Road",
        "Sidewalk",
        "Parking",
        "Rail Track",
        "Person",
        "Rider",
        "Car",
        "Truck",
        "Bus",
        "On Rails",
        "Motorcycle",
        "Bicycle",
        "Caravan",
        "Trailer",
        "Building",
        "Wall",
        "Fence",
        "Guard Rail",
        "Bridge",
        "Tunnel",
        "Pole",
        "Pole Group",
        "Traffic Sign",
        "Traffic Light",
        "Vegetation",
        "Terrain",
        "Sky",
        "Ground",
        "Dynamic",
        "Static",
        "Plane_0 (1)"
    ] + ["Plane_{}".format(i) for i in range(100)]

    for key, value in res.items():
        print("{0}: {1}".format(key, list(map(lambda x: label_names[x], value))))
