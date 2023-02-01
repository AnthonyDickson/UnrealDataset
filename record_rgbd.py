import argparse
import json
import os
import shutil
import time
from io import BytesIO
from multiprocessing.pool import ThreadPool
from typing import List

import imageio
import numpy as np
from pynput import keyboard
from scipy import interpolate
from scipy.spatial.transform import Rotation, Slerp
from unrealcv import Client

from UnrealDatasetInfo import UnrealDatasetInfo


def read_trajectory(filename):
    with open(filename) as f:
        trajectory = json.load(f)

    def convert_line(line):
        transform = list(map(float, line.split(' ')))
        position, rotation = transform[:3], transform[3:]

        return [*position, *rotation]

    parsed_trajectory = list(map(convert_line, trajectory))

    return np.array(parsed_trajectory)


def read_npy(res):
    return np.load(BytesIO(res))


def interpolate_poses(trajectory, frame_delay):
    frame_times = np.arange(0.0, len(trajectory) - 1, frame_delay)

    translation_vectors, rotation_vectors = trajectory[:, :3], trajectory[:, 3:]

    translation_lerp = interpolate.interp1d(np.arange(len(translation_vectors)), translation_vectors, axis=0)
    interpolated_translations = translation_lerp(frame_times)

    key_rots = rotation_vectors
    rot_offsets = np.zeros_like(rotation_vectors)

    # Offset rotations so that camera in Unreal will take shortest path to next camera pose.
    # This prevents the camera suddenly spinning in the 'wrong' direction.
    for axis in range(0, 2):
        curr_offset = 0.0

        for i, rot in enumerate(key_rots):
            if i > 0:
                prev_yaw = key_rots[i - 1, axis]
                curr_yaw = rot[axis]

                has_shorter_path = abs(prev_yaw - curr_yaw) > abs(360 - abs(prev_yaw - curr_yaw))
                if has_shorter_path:
                    if prev_yaw > curr_yaw:
                        curr_offset += 360.0
                    else:
                        curr_offset -= 360.0

            rot_offsets[i, axis] = curr_offset

    offset_key_rots = key_rots + rot_offsets

    rotations_lerp = interpolate.interp1d(np.arange(len(rotation_vectors)), offset_key_rots, axis=0)
    interpolated_rotations = rotations_lerp(frame_times)

    interpolated_trajectory = np.array([
        [*translation, *rotation]
        for (translation, rotation) in zip(
            interpolated_translations,
            np.around(interpolated_rotations, 2)
        )
    ])

    return interpolated_trajectory


def get_frame_data(client: Client, interpolated_trajectory, frame_delay=1. / 30.):
    frames = []

    for i, pose in enumerate(interpolated_trajectory):
        start = time.time()

        x, y, z, pitch, yaw, roll = pose

        client.request(f"vset /camera/0/pose {x} {y} {z} {pitch} {yaw} {roll}")

        color_res, depth_res = client.request_batch(['vget /camera/0/lit npy', 'vget /camera/0/depth npy'])
        frames.append([color_res, depth_res])

        elapsed = time.time() - start
        time_to_sleep = max(0.0, frame_delay - elapsed)
        print(f"Captured frame {i + 1} of {len(interpolated_trajectory)} in {elapsed:.3f} seconds.")

        time.sleep(time_to_sleep)

    fov = float(client.request('vget /camera/0/horizontal_fieldofview'))

    return fov, frames


def convert_to_depth_to_plane(depth_map, f):
    h, w = depth_map.shape[:2]
    i_c = float(h) / 2 - 1
    j_c = float(w) / 2 - 1
    columns, rows = np.meshgrid(np.linspace(0, w - 1, num=w), np.linspace(0, h - 1, num=h))
    distance_from_center = ((rows - i_c) ** 2 + (columns - j_c) ** 2) ** 0.5
    depth_map_plane = depth_map / (1 + (distance_from_center / f) ** 2) ** 0.5

    return depth_map_plane


def write_frame_data_to_disk(frames, colour_dir, depth_dir, focal_length, max_depth=10.0, invalid_depth_value=0.0,
                             overwrite_ok=False):
    if overwrite_ok:
        if os.path.exists(colour_dir):
            shutil.rmtree(colour_dir)

        if os.path.exists(depth_dir):
            shutil.rmtree(depth_dir)

    os.makedirs(colour_dir)
    os.makedirs(depth_dir)

    def write_frame_data(i, frame_data):
        color_frame_buffer, depth_map_buffer = frame_data
        color = read_npy(color_frame_buffer)
        color = color[:, :, :3]
        color = np.ascontiguousarray(color)

        depth = read_npy(depth_map_buffer)
        # UnrealCV reads depth as depth to the center of the camera, however using the depth maps as is results in
        # warped surfaces. To get back straight walls etc. we need to convert the depth map to use depth values to the
        # camera plane (Z-plane).
        depth = convert_to_depth_to_plane(depth, focal_length)
        # Clip depth values.
        depth[depth > max_depth] = invalid_depth_value
        # Convert depth values to mm.
        depth = 1000 * depth
        depth = depth.astype(np.uint16)

        filename = f"{i:06d}"
        color_path = os.path.join(colour_dir, f"{filename}.png")
        depth_path = os.path.join(depth_dir, f"{filename}.png")

        print(f"Saving color and depth frame {i + 1:03,d} of {len(frames):03,d} to: {color_path} AND {depth_path}...")
        imageio.imwrite(color_path, color)
        imageio.imwrite(depth_path, depth)

    pool = ThreadPool(processes=8)
    pool.starmap(write_frame_data, list(enumerate(frames)))


def convert_unreal_trajectory(trajectory: List[str]) -> np.ndarray:
    """
    Convert a list of poses from UnrealCV into a single NumPy array.
    :param trajectory: The list of 6-vector poses as a string where the first three components are the coordinates of
        the camera, and the last components make up the Euler angles rotation of the camera.
    :return: The converted trajectory.
    """
    translation_vectors = []
    rotation_vectors = []

    for line in trajectory:
        x, y, z, pitch, yaw, roll = map(float, line.split(' '))
        translation_vectors.append([x, y, z])
        rotation_vectors.append([pitch, yaw, roll])

    t = np.asarray(translation_vectors)
    r = np.asarray(rotation_vectors)

    return np.hstack((t, r))


def main(output_path, client_url='localhost', client_port=8888, max_depth=10.0, invalid_depth_value=0.0,
         fps=30.0, intrinsics_filename='camera.txt', trajectory_filename='trajectory.txt',
         colour_folder='colour', depth_folder='depth', overwrite_ok=False):
    trajectory = []

    client = Client((client_url, client_port))
    client.connect()

    def _onkeypress(key):
        try:
            k = key.char  # single-char keys
        except:
            k = key.name  # other keys
        # check for quit condition
        if key == keyboard.Key.esc:
            print("LOG INFO: Time to quit")
            return False

        if k == 'p':  # On pressing 'p', capture pose.
            print("Capturing position!")
            pose = client.request('vget /camera/0/pose')
            trajectory.append(pose)
            print("Pose: {} captured!".format(pose))
        elif k == 'r':
            print("Recording RGB-D sequence... Try not to move your mouse.")
            return False

        return True

    keyboard_listener = keyboard.Listener(on_press=_onkeypress)
    keyboard_listener.start()
    keyboard_listener.join()

    print(f"LOG INFO: Created trajectory of length {len(trajectory)}.")

    if len(trajectory) == 0:
        print("LOG INFO: Zero length trajectory, quitting...")
        return

    os.makedirs(output_path, exist_ok=overwrite_ok)

    frame_delay = 1. / fps

    print(f"Length of Camera Trajectory: {len(trajectory)}")
    print(f"Capture Frame Rate: {fps}")
    print(f"Capture Frame Delay (s): {frame_delay:.4f}")
    print(f"Total Interpolated Poses: {fps * (len(trajectory) - 1)}")

    trajectory_np = convert_unreal_trajectory(trajectory)
    interpolated_trajectory = interpolate_poses(trajectory_np, frame_delay)

    fov, frames = get_frame_data(client, interpolated_trajectory, frame_delay)
    client.disconnect()

    output_trajectory = np.hstack((
        Rotation.from_euler('xyz', interpolated_trajectory[:, 3:], degrees=True).as_quat(),
        # Translation units in Unreal is cm, convert it to meters to it's the same as depth.
        interpolated_trajectory[:, :3] / 100.0
    ))

    trajectory_output_file = os.path.join(output_path, trajectory_filename)
    print(f"Saving trajectory to {trajectory_output_file}...")
    np.savetxt(trajectory_output_file, output_trajectory)

    height, width = read_npy(frames[0][0]).shape[:2]
    cu, cv = width / 2, height / 2
    fov_radians = fov * np.pi / 180
    # Reference for this formula: https://purehost.bath.ac.uk/ws/portalfiles/portal/228418730/Bertel_PhD_thesis_169465296_compressed_Redacted.pdf
    f = width / (2 * np.tan(fov_radians / 2))

    K = np.array([[f, 0., cu],
                  [0., f, cv],
                  [0., 0., 1.]])

    print(f"Writing frames to {output_path}...")
    colour_dir = os.path.join(output_path, colour_folder)
    depth_dir = os.path.join(output_path, depth_folder)
    write_frame_data_to_disk(frames, colour_dir, depth_dir, f, max_depth, invalid_depth_value,
                             overwrite_ok=overwrite_ok)

    camera_params_file = os.path.join(output_path, intrinsics_filename)
    print(f"Saving camera intrinsics to {camera_params_file}...")
    np.savetxt(camera_params_file, K)

    info = UnrealDatasetInfo(width=width, height=height, num_frames=len(frames), fps=fps, max_depth=max_depth,
                             invalid_depth_value=invalid_depth_value, intrinsics_filename=intrinsics_filename,
                             trajectory_filename=trajectory_filename, colour_folder=colour_folder,
                             depth_folder=depth_folder)

    info_file = os.path.join(output_path, 'info.json')
    info.save_json(info_file)

    print(f"Saved dataset info to {info_file}")

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='rgbd_dataset')
    parser.add_argument('--fps', default=30, type=int)
    parser.add_argument('--max_depth', help='The value the depth values are clipped to.', default=10.0, type=float)
    parser.add_argument('--invalid_depth_value', help='The value to use for clipped depth values.', default=0.0,
                        type=float)
    parser.add_argument('--overwrite_ok', action='store_true',
                        help='Whether to overwrite files if the output path already exists.')

    parser.add_argument('--client_url', default='localhost')
    parser.add_argument('--client_port', type=int, default=8888)

    args = parser.parse_args()

    output_path = args.output_path
    max_depth = args.max_depth
    invalid_depth_value = args.invalid_depth_value
    fps = args.fps
    client_url = args.client_url
    client_port = args.client_port
    overwrite_ok = args.overwrite_ok

    main(output_path, client_url, client_port, max_depth, invalid_depth_value, overwrite_ok=overwrite_ok, fps=fps)
