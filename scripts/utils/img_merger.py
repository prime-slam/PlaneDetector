import os

from PIL import Image

if __name__ == "__main__":
    width = 640
    height = 480
    padding_x = 10
    padding_y = 10
    res = Image.new('RGB', (width * 9 + 9 * padding_x, height * 6 + 6 * padding_y))
    res.paste((256, 256, 256), [0, 0, res.size[0], res.size[1]])
    matrix = [
        ["rgbd_qual-0009.png", "rgbd_qual-0008.png", "rgbd_qual-0013.png", "rgbd_qual-0016.png", "rgbd_qual-0026.png", "rgbd_qual-0032.png",
         "rgbd_qual-0038.png", "rgbd_qual-0040.png", "rgbd_qual-0046.png", ],
        ["rgbd_qual-0002.png", "rgbd_qual-0003.png", "rgbd_qual-0014.png", "rgbd_qual-0017.png", "rgbd_qual-0025.png", "rgbd_qual-0031.png",
         "rgbd_qual-0037.png", "rgbd_qual-0051.png", "rgbd_qual-0045.png", ],
        ["rgbd_qual-0011.png", "rgbd_qual-0006.png", "rgbd_qual-0015.png", "rgbd_qual-0021.png", "rgbd_qual-0027.png", "rgbd_qual-0039.png",
         "rgbd_qual-0033.png", "rgbd_qual-0047.png", "rgbd_qual-0041.png", ],
        ["rgbd_qual-0001.png", "rgbd_qual-0004.png", "rgbd_qual-0054.png", "rgbd_qual-0020.png", "rgbd_qual-0022.png", "rgbd_qual-0028.png",
         "rgbd_qual-0034.png", "rgbd_qual-0042.png", "rgbd_qual-0048.png", ],
        ["rgbd_qual-0012.png", "rgbd_qual-0007.png", "rgbd_qual-0053.png", "rgbd_qual-0019.png", "rgbd_qual-0023.png", "rgbd_qual-0029.png",
         "rgbd_qual-0035.png", "rgbd_qual-0049.png", "rgbd_qual-0043.png", ],
        ["rgbd_qual-0010.png", "rgbd_qual-0005.png", "rgbd_qual-0052.png", "rgbd_qual-0018.png", "rgbd_qual-0024.png", "rgbd_qual-0030.png",
         "rgbd_qual-0036.png", "rgbd_qual-0050.png", "rgbd_qual-0044.png", ],
    ]

    for x in range(9):
        for y in range(6):
            path = os.path.join("C:\\Users\\dimaj\\Desktop\\rgbd_qual", matrix[y][x])
            img = Image.open(path)
            res.paste(img, (x * width + x * padding_x, y * height + y * padding_y))

    res.save("out.jpg")
