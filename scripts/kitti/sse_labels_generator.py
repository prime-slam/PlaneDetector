import sys

from src.utils.colors import get_random_color

if __name__ == "__main__":
    amount_to_generate = int(sys.argv[1])
    for i in range(amount_to_generate):
        color = get_random_color()
        hex_color_str = "#{0:02X}{1:02X}{2:02X}".format(color[0], color[1], color[2])
        print('{{"label": "Plane_{0}", "color": "{1}", "icon": "Road"}},'.format(i, hex_color_str))
