import argparse
import json
import sys

from pynput import keyboard
from unrealcv import Client


def save_to_file(filename, trajectory):
    if len(trajectory) != 0:
        with open(filename, 'w') as f:
            json.dump(trajectory, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='camera-trajectory.json')
    parser.add_argument('--client_url', default='localhost')
    parser.add_argument('--client_port', type=int, default=8888)
    args = parser.parse_args()

    trajectory = []

    client = Client(('localhost', args.client_port))
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
        try:
            if k == 'p':  # On pressing 'p', capture position.
                print("Capturing position!")
                pose = client.request('vget /camera/0/pose')
                trajectory.append(pose)
                print("Position: {} captured!".format(pose))
                # You can do the rest.
        except Exception as e:
            raise e

    keyboard_listener = keyboard.Listener(on_press=_onkeypress)
    keyboard_listener.start()
    keyboard_listener.join()

    print(f"LOG INFO: Saving trajectory of length {len(trajectory)} to {args.filename}.")
    save_to_file(args.filename, trajectory)
    client.disconnect()
    print("LOG INFO: Quit command completed")


if __name__ == "__main__":
    sys.exit(main())
