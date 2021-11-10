"""Convert dataset into the format that Kinect Fusion and related methods require (everything in one directory >:| )."""
import argparse
import cv2
import numpy as np
import os
import shutil

from UnrealDatasetInfo import UnrealDatasetInfo

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert dataset into the format that Kinect Fusion and related methods "
                                     "require (everything in one directory >:| ).")
    parser.add_argument('--base_path', default='/path/to/source/dataset')
    parser.add_argument('--output_path', default='/path/to/save/output')
    args = parser.parse_args()

    base_path = os.path.abspath(args.base_path)
    output_path = os.path.abspath(args.output_path)

    info = UnrealDatasetInfo.from_json(os.path.join(base_path, 'info.json'))

    cam_intr = np.loadtxt(os.path.join(base_path, info.intrinsics_filename))
    camera_trajectory = np.loadtxt(os.path.join(base_path, info.trajectory_filename))

    color_path = os.path.join(base_path, info.colour_folder)
    color_filenames = sorted(os.listdir(color_path))
    depth_path = os.path.join(base_path, info.depth_folder)
    depth_filenames = sorted(os.listdir(depth_path))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, (color_filename, depth_filename, pose) in enumerate(
            zip(color_filenames, depth_filenames, camera_trajectory)):
        frame_name = "frame-{:06d}".format(i)
        color_output_path = os.path.join(output_path, f"{frame_name}.color.jpg")
        depth_output_path = os.path.join(output_path, f"{frame_name}.depth.png")
        pose_output_path = os.path.join(output_path, f"{frame_name}.pose.txt")

        input_colour_path = os.path.join(color_path, color_filename)
        print(f"Copying {input_colour_path} -> {color_output_path}")
        shutil.copyfile(input_colour_path, color_output_path)

        input_depth_path = os.path.join(depth_path, depth_filename)
        print(f"Copying {input_depth_path} -> {depth_output_path}")
        shutil.copyfile(input_depth_path, depth_output_path)

        pose_mat = np.eye(4, dtype=np.float32)
        pose_mat[0:3, 0:3] = cv2.Rodrigues(pose[:3])[0]
        pose_mat[0:3, -1] = pose[-3:].reshape((1, -1))

        np.savetxt(pose_output_path, pose_mat)
        print("Saved data for frame {:06d}...".format(i))

    depth_scaling_factor = np.iinfo(np.uint16 if info.is_16bit_depth else np.uint8).max / info.max_depth

    info_txt = f"""m_versionNumber = 4
m_sensorName = UNREAL
m_colorWidth = {info.width}
m_colorHeight = {info.height}
m_depthWidth = {info.width}
m_depthHeight = {info.height}
m_depthShift = {depth_scaling_factor}
m_calibrationColorIntrinsic = {' '.join(map(str, cam_intr.astype(int).ravel()))} 
m_calibrationColorExtrinsic = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 
m_calibrationDepthIntrinsic = {' '.join(map(str, cam_intr.astype(int).ravel()))} 
m_calibrationDepthExtrinsic = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 
m_frames.size = {info.num_frames}
"""

    with open(os.path.join(output_path, "info.txt"), 'w') as f:
        f.writelines(info_txt)
