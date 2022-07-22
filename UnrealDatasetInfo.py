import json


class UnrealDatasetInfo:
    def __init__(self, width, height, num_frames, fps=30.0, max_depth=10.0, invalid_depth_value=0.0,
                 intrinsics_filename='camera.txt', trajectory_filename='trajectory.txt', colour_folder='colour',
                 depth_folder='depth'):
        """
        :param width: The width in pixels of the colour frames and depth maps.
        :param height: The height in pixels of the colour frames and depth maps.
        :param num_frames: The number of frames in the dataset.
        :param fps: The framerate of the captured video.
        :param max_depth: The maximum depth value allowed in a depth map.
        :param invalid_depth_value: The values used to indicate invalid (e.g. missing) depth.
        :param intrinsics_filename: The name of camera parameters file.
        :param trajectory_filename: The name of the camera pose file.
        :param colour_folder: The name of the folder that contains the colour frames.
        :param depth_folder: The name of the folder that contains the depth maps.
        """
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.max_depth = max_depth
        self.invalid_depth_value = invalid_depth_value
        self.intrinsics_filename = intrinsics_filename
        self.trajectory_filename = trajectory_filename
        self.colour_folder = colour_folder
        self.depth_folder = depth_folder

    def save_json(self, fp):
        if isinstance(fp, str):
            with open(fp, 'w') as f:
                json.dump(self.__dict__, f)
        else:
            json.dump(self.__dict__, fp)

    @staticmethod
    def from_json(fp):
        if isinstance(fp, str):
            with open(fp, 'r') as f:
                data = json.load(f)
        else:
            data = json.load(fp)

        return UnrealDatasetInfo(**data)
