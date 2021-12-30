import numpy as np


def get_random_color():
    return tuple([int(x) for x in np.random.choice(range(256), size=3)])


def normalize_color(color):
    return tuple([channel / 255 for channel in color])


def denormalize_color(color):
    return tuple([channel * 255 for channel in color])


def get_random_normalized_color():
    return normalize_color(get_random_color())


def color_to_string(color_arr):
    # np.array2string is critically slow
    return "{0},{1},{2}".format(color_arr[0], color_arr[1], color_arr[2])


def color_from_string(color_str):
    return np.fromiter((float(channel_str) for channel_str in color_str.decode('UTF-8').split(',')), dtype=np.float64)
