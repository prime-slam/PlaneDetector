import os
import sys
import shutil

if __name__ == "__main__":
    input_dir = sys.argv[1]
    start_ind = int(sys.argv[2])
    end_ind = int(sys.argv[3])
    step = int(sys.argv[4])
    output_dir = sys.argv[5]

    image_filepaths = [
        os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith(".jpg")
    ]

    image_filepaths = sorted(image_filepaths, key=lambda x: int(os.path.split(x)[-1].split(".")[0][6:]))
    for index, image_path in enumerate(image_filepaths):
        if start_ind <= index <= end_ind and (index - start_ind) % step == 0:
            image_filename = os.path.split(image_path)[-1]
            shutil.copyfile(image_path, os.path.join(output_dir, image_filename))
