from src.loaders.tum import TumDataset


class TumIclDataset(TumDataset):
    def provide_filenames(self, rgb_path, depth_path):
        rgb_filenames, depth_filenames = super(TumIclDataset, self).provide_filenames(
            rgb_path,
            depth_path
        )
        normalized_rgb_filenames = self.normalize_filenames(rgb_filenames)
        normalized_depth_filenames = self.normalize_filenames(depth_filenames)

        return normalized_rgb_filenames, normalized_depth_filenames

    def normalize_filenames(self, filenames):
        max_filename_len = max([len(filename[:-4]) for filename in filenames])
        normalized_filenames = [
            "0" * (max_filename_len - len(filename[:-4])) + filename for filename in filenames
        ]

        return normalized_filenames
