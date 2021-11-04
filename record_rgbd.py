import argparse
import json
import os
import time
from io import BytesIO

import imageio
import numpy as np
from scipy import interpolate
from scipy.spatial.transform import Rotation
from unrealcv import Client


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


def interpolate_poses(translation_vectors, rotation_vectors, frame_delay):
    frame_times = np.arange(0.0, len(trajectory) - 1, frame_delay)

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


def get_frame_data(interpolated_trajectory, frame_delay=1. / 30., client_url='localhost', client_port=8888):
    client = Client((client_url, client_port))
    client.connect()

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

    client.disconnect()

    return fov, frames


def write_frame_data_to_disk(frames, output_path, max_depth=10.0, invalid_depth_value=0.0):
    print(f"Writing frames to {output_path}...")
    color_dir = os.path.join(output_path, 'colour')
    depth_dir = os.path.join(output_path, 'depth')
    os.makedirs(color_dir)
    os.makedirs(depth_dir)

    for i, (color_frame_buffer, depth_map_buffer) in enumerate(frames):
        color = read_npy(color_frame_buffer)
        color = color[:, :, :3]
        color = np.ascontiguousarray(color)

        depth = read_npy(depth_map_buffer)
        # Clip depth values.
        depth[depth > max_depth] = invalid_depth_value
        # Normalize to [0.0, 1.0]
        depth /= max_depth
        # Expand values to take up uint16 range and convert to uint16
        depth = np.iinfo(np.uint16).max * depth
        depth = depth.astype(np.uint16)

        filename = f"{i:03,d}"
        color_path = os.path.join(color_dir, f"{filename}.jpg")
        depth_path = os.path.join(depth_dir, f"{filename}.png")

        print(f"Saving color and depth frame {i + 1:03,d} of {len(frames):03,d} to: {color_path} AND {depth_path}...")
        imageio.imwrite(color_path, color)
        imageio.imwrite(depth_path, depth)


def main(trajectory, output_path, max_depth=10.0, invalid_depth_value=0.0, fps=30.0, client_url='localhost', client_port=8888):
    os.makedirs(output_path)

    frame_delay = 1. / fps

    print(f"Length of Camera Trajectory: {len(trajectory)}")
    print(f"Capture Frame Rate: {fps}")
    print(f"Capture Frame Delay (s): {frame_delay:.4f}")
    print(f"Total Interpolated Poses: {fps * (len(trajectory) - 1)}")

    translations, rotations = trajectory[:, :3], trajectory[:, 3:]
    interpolated_trajectory = interpolate_poses(translations, rotations, frame_delay)

    fov, frames = get_frame_data(interpolated_trajectory, frame_delay, client_url, client_port)

    translation_vectors = interpolated_trajectory[:, :3]
    rotation_vectors = Rotation.from_euler('xyz', interpolated_trajectory[:, 3:], degrees=True).as_rotvec()
    output_trajectory = np.hstack((rotation_vectors, translation_vectors))

    trajectory_output_file = os.path.join(output_path, 'trajectory.txt')
    print(f"Saving trajectory to {trajectory_output_file}...")
    np.savetxt(trajectory_output_file, output_trajectory)

    write_frame_data_to_disk(frames, output_path, max_depth, invalid_depth_value)

    height, width = read_npy(frames[0][0]).shape[:2]
    cu, cv = width / 2, height / 2
    f = width / (2 * np.tan(fov * np.pi / 360))

    K = np.array([[f, 0., cu],
                  [0., f, cv],
                  [0., 0., 1.]])

    camera_params_file = os.path.join(output_path, 'camera.txt')
    print(f"Saving camera intrinsics to {camera_params_file}...")
    np.savetxt(camera_params_file, K)

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='camera-trajectory.json')
    parser.add_argument('--output_path', default='rgbd_dataset')
    parser.add_argument('--fps', default=30, type=int)
    parser.add_argument('--max_depth', help='The value the depth values are clipped to.', default=10.0, type=float)
    parser.add_argument('--invalid_depth_value', help='The value to use for clipped depth values.', default=0.0,
                        type=float)

    parser.add_argument('--client_url', default='localhost')
    parser.add_argument('--client_port', type=int, default=8888)

    args = parser.parse_args()

    trajectory = read_trajectory(args.filename)
    output_path = args.output_path
    max_depth = args.max_depth
    invalid_depth_value = args.invalid_depth_value
    fps = args.fps
    client_url = args.client_url
    client_port = args.client_port

    main(trajectory, output_path, max_depth, invalid_depth_value, fps, client_url, client_port)
