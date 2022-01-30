import sys

import numpy as np


def color_to_string(color_arr):
    # np.array2string is critically slow
    return "{0},{1},{2}".format(color_arr[0], color_arr[1], color_arr[2])


UNSEGMENTED_PCD_COLOR = [0, 0, 0]  # [0.5, 0.5, 0.5]
produced_colors_set = {color_to_string(UNSEGMENTED_PCD_COLOR)}


def get_random_color():
    random_color = np.asarray(UNSEGMENTED_PCD_COLOR)
    while color_to_string(random_color) in produced_colors_set:
        random_color = np.asarray([int(x) for x in np.random.choice(range(256), size=3)])

    produced_colors_set.add(color_to_string(random_color))

    return random_color


if __name__ == "__main__":
    amount_to_generate = int(sys.argv[1])
    for i in range(amount_to_generate):
        color = get_random_color()
        hex_color_str = "#{0:02X}{1:02X}{2:02X}".format(color[0], color[1], color[2])
        print('      {{"label": "Plane_{0}", "color": "{1}", "icon": "Road"}},'.format(i, hex_color_str))
