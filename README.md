# UnrealDataset

# Getting Started
## Unreal Environment
See https://unrealcv.org
## Python Environment
Set up your Python environment via the one of the following options:
1. Set up via Conda:
    ```shell
    conda env create -f environment.yml
    ```
2. Set up via PiP:
    ```shell
    pip install scipy numpy imageio pynput unrealcv
    ```

# Creating Datasets
1. Open up Unreal Engine with the scene you want to record and make sure the UnrealCV plugin is active.
   1. (optional) Open the output console so you can see the UnrealCV logs.
2. Click the play button to enter live preview.
3. Run the script `record_rgbd.py`.
4. Move the camera in the Unreal Engine editor and press `p` to record the camera's pose.
5. Repeat the previous step until you have all the poses you want.
6. Press `r` to start recording the RGB-D dataset. Try not to move your mouse during recording, or it will affect the camera pose and captured RGB-D frames.