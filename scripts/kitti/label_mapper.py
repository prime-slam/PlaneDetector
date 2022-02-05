import os

import open3d as o3d

if __name__ == "__main__":
    path_dir = "debug_map"
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
        'part_-148_1_-82_67_u58.pcd': (1, 60),
        'part_-298_-148_67_217_u59.pcd': (61, 119),
        'part_1_151_217_367_u50.pcd': (120, 222)
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
        'part_-148_1_-82_67_u58.pcd': [],
        'part_-298_-148_67_217_u59.pcd': [],
        'part_1_151_217_367_u50.pcd': []
    }

    for filename in filenames:
        label_id = int(filename.split(".")[0])
        for key, value in parts.items():
            min_val, max_val = value
            if min_val <= label_id <= max_val:
                res[key].append(label_id - min_val)

    label_names = [
        "VOID",
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
        "Plane_0"
    ] + ["Plane_{}".format(i) for i in range(100)]

    for key, value in res.items():
        print("{0}: {1}".format(key, list(map(lambda x: label_names[x], value))))

# 8, 11, 13, 18, 34, 41, 63, 71, 73, 160, 179, 190, 243, 244, 249, 324, 349, 369, 410, 430, 449, 467 -- before eye
# 21, 117, 142, 151, 183!!!, 215 - after eye