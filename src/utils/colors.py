import numpy as np


def color_to_string(color_arr):
    # np.array2string is critically slow
    return "{0},{1},{2}".format(color_arr[0], color_arr[1], color_arr[2])


def color_from_string(color_str):
    return np.fromiter(
        (float(channel_str) for channel_str in color_str.split(",")), dtype=np.float64
    )


def normalize_color(color):
    return np.asarray([channel / 255 for channel in color])


def normalize_color_arr(color_arr):
    return color_arr / 255


def denormalize_color(color):
    return np.round(np.asarray([channel * 255 for channel in color]))


def denormalize_color_arr(color_arr):
    return color_arr * 255


UNSEGMENTED_PCD_COLOR = [0, 0, 0]  # [0.5, 0.5, 0.5]
UNSEGMENTED_PCD_COLOR_NORMALISED = normalize_color(UNSEGMENTED_PCD_COLOR)
produced_colors_set = {color_to_string(UNSEGMENTED_PCD_COLOR)}


def get_random_color():
    random_color = np.asarray(UNSEGMENTED_PCD_COLOR)
    while color_to_string(random_color) in produced_colors_set:
        random_color = np.asarray(
            [int(x) for x in np.random.choice(range(256), size=3)]
        )

    produced_colors_set.add(color_to_string(random_color))

    return random_color


def get_random_normalized_color():
    return normalize_color(get_random_color())
