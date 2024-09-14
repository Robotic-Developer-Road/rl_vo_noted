# Reinforcement Learning Meets Visual Odometry

<p align="center">
 <a href="https://youtu.be/pt6yPTdQd6M">
  <img src="doc/thumbnail.png" alt="youtube_video" width="800"/>
 </a>
</p>

This is the code for the ECCV24 paper **Reinforcement Learning Meets Visual Odometry**
([PDF](https://rpg.ifi.uzh.ch/docs/ECCV24_Messikommer.pdf)) by [Nico Messikommer*](https://messikommernico.github.io/), [Giovanni Cioffi*](https://giovanni-cioffi.netlify.app/), [Mathias Gehrig](https://magehrig.github.io/), and [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html).
For an overview of our method, check out our [video](https://youtu.be/pt6yPTdQd6M).

If you use any of this code, please cite the following publication:

```bibtex
@Article{Messikommer24eccv,
  author  = {Nico Messikommer* and Giovanni Cioffi* and Mathias Gehrig and Davide Scaramuzza},
  title   = {Reinforcement Learning Meets Visual Odometry},
  journal = {European Conference on Computer Vision. (ECCV)},
  year    = {2024},
}
```

If you use the SVO library, please do not forget to cite the following publications:

```bibtex
@inproceedings{Forster2014ICRA,
  author = {Forster, Christian and Pizzoli, Matia and Scaramuzza, Davide},
  title = {{SVO}: Fast Semi-Direct Monocular Visual Odometry},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year = {2014}
}
```

```bibtex
@ARTICLE{7782863,
  author={Forster, Christian and Zhang, Zichao and Gassner, Michael and Werlberger, Manuel and Scaramuzza, Davide},
  journal={IEEE Transactions on Robotics}, 
  title={SVO: Semidirect Visual Odometry for Monocular and Multicamera Systems}, 
  year={2017},
  volume={33},
  number={2},
  pages={249-265},
  doi={10.1109/TRO.2016.2623335}}
```

## Abstract

Visual Odometry (VO) is essential to downstream mobile robotics and augmented/virtual reality tasks. 
Despite recent advances, existing VO methods still rely on heuristic design choices that require several weeks of hyperparameter tuning by human experts, hindering generalizability and robustness. 
We address these challenges by reframing VO as a sequential decision-making task and applying Reinforcement Learning (RL) to adapt the VO process dynamically. 
Our approach introduces a neural network, operating as an agent within the VO pipeline, to make decisions such as keyframe and grid-size selection based on real-time conditions. 
Our method minimizes reliance on heuristic choices using a reward function based on pose error, runtime, and other metrics to guide the system. 
Our RL framework treats the VO system and the image sequence as an environment, with the agent receiving observations from keypoints, map statistics, and prior poses. 
Experimental results using classical VO methods and public benchmarks demonstrate improvements in accuracy and robustness, validating the generalizability of our RL-enhanced VO approach to different scenarios. 
We believe this paradigm shift advances VO technology by eliminating the need for time-intensive parameter tuning of heuristics.

## Installation
Below are the instructions to run the RL framework with SVO.

### RL Framework

Go to the directory of the repo (vo_rl/) and run the follwing command:

`docker build -t vo_rl .`

To launch the contain of the built image, first, the paths inside the `launch_container.sh` needs to be adjusted to point to the code, data and log directory.
Once the paths are changed, the following command can be used:

`bash launch_container.sh`

The following commands can be used to install the SVO library:

`cd svo-lib`

`mkdir build && cd build`

`cmake .. && make`

### SVO Library
In addition to the RL framework, there is also a separate dockerfile inside the `svo-lib` directory, which can be used to build the SVO library in a similar way as the RL framework.

## Datasets
### TartanAir
The TartanAir dataset can be downloaded here:
[TartanAir](https://theairlab.org/tartanair-dataset/)
To speed up the dataloading, we convert the RGB images of the `image_left` directory to grayscale images.

The dataset should have the following structure:
```
TartanAir/
├── abandonedfactory/
│   ├── Easy/
│   │   ├── P000/
│   │   │   ├── image_left_gray/
│   │   │   │   ├── 000000_left.jpg
│   │   │   │   ├── ...
│   │   │   ├── pose_left.txt
```

### EuRoC
The EuRoC dataset can be downloaded here:
[EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)

The dataset should have the following structure:
```
EuRoC/
├── MH_01_easy/
│   ├── mav0/
│   │   ├── cam0/
│   │   │   ├── data.csv
│   │   │   ├── data/
│   │   │   │   ├── 1403636579763555584.png
│   │   │   │   ├── ...
│   │   ├── state_groundtruth_estimate0/
│   │   │   ├── data.csv
│   │   │   ├── sensor.yaml
```

### TUM-RGBD
The TUM-RGBD dataset can be downloaded here:
[TUM-RGBD](https://vision.in.tum.de/data/datasets/rgbd-dataset/download)

The dataset should have the following structure:
```
TUM-RGBD/
├── rgbd_dataset_freiburg1_desk/
│   ├── groundtruth.txt
│   ├── rgb.txt
│   ├── rgb/
│   │   ├── 1305031452.791720.png
│   │   ├── ...
```

## Training
In a first step, set all the necessary training parameters, the path to the dataset, config and logging directory in the `config` folder.
To use wandb for logging, set the `wandb_logging` variable to `True` and specify the wandb tags and group for the specific training run in `config.py` file.
The wandb initialization is done inside the `ppo.py` file, where the `<entity>` shoud be replaced before running.
To train the model inside the docker container, run the following command by replacing <GPU_ID> with the GPU ID to be used:

`CUDA_VISIBLE_DEVICES=<GPU_ID> python train.py`

## Testing
Before testing the model, set all the necessary testing parameters, the path to the dataset, config and logging directory in the `config_eval.py` file.
To test the model inside the docker container, run the following command by replacing <GPU_ID> with the GPU ID to be used:

`CUDA_VISIBLE_DEVICES=<GPU_ID> python play.py`

To evaluate the performance of the different runs, run the `/evaluation/evaluate_runs.py`.
First, add the paths pointing to the results of the `play.py` script to the corresponding method in the `METHODS` dict inside `evaluate_runs.py`.
Additionally, also specify the `OUT_DIR` in the `evaluate_runs.py` file.
Then, run the following command from the level of the `vo_rl` directory to evaluate the runs:

`python -m evaluation.evaluate_runs`


## Usage


### Config
Only `Tartanair` is supported for train now, `euroc` and `tum` are not supported yet. because `sel.mode` in the dataset_loader.py

- Tartanair: just need to download image_left.zip 

### Train
```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train.py
```

- DEVICE0: Nvidia GeForce RTX 3090
- DEVICE1: UHD graphics 630

### Problem

check param of `image_width` of `svo_calib_file` your  in `config.yaml` file

#### 1. `Check failed: img.cols == static_cast<int>(cam_->imageWidth()) (640 vs. 752)`
```shell
F0907 16:54:42.519456  1385 frame.cpp:76] Check failed: img.cols == static_cast<int>(cam_->imageWidth()) (640 vs. 752) 
```
- run
```shell
ruoxi@robot2go:~/rl_vo$ bash launch_container.sh 
```

- lauch focker container
```shell
non-network local connections being added to access control list
current path: /home/ruoxi

==========
== CUDA ==
==========

CUDA Version 12.1.0

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

*************************
** DEPRECATION NOTICE! **
*************************
THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
    https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md

groups: cannot find name for group ID 1000
```

- run
```shell
ruoxi@robot2go:~/vo_rl$ CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train.py
```

- output
```shell
Matplotlib created a temporary cache directory at /tmp/matplotlib-chasjv53 because the default path (/home/ruoxi/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
loaded 1 cameras
  name = cam
  size = [752, 480]
  Projection = Pinhole
  Focal length = (383.013, 382.392)
  Principal point = (344.706, 244.327)
  Distortion: Atan(0.932)
..............
loaded 1 cameras
  name = cam
  size = [752, 480]
  Projection = Pinhole
  Focal length = (383.013, 382.392)
  Principal point = (344.706, 244.327)
  Distortion: Atan(0.932)
Using cuda:0 device
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 1
wandb: You chose 'Create a W&B account'
wandb: Create an account here: https://wandb.ai/authorize?signup=true
wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit: 
wandb: Appending key for api.wandb.ai to your netrc file: /home/ruoxi/.netrc
wandb: ERROR Unable to read /home/ruoxi/.netrc
wandb: Tracking run with wandb version 0.17.9
wandb: Run data is saved locally in /home/ruoxi/vo_rl/logs/log_voRL/training/<group>/Sep07_03-08-31_<Tag>/wandb/run-20240907_031249-0mlj4f6f
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run legendary-jazz-1
wandb: ⭐️ View project at https://wandb.ai/sujit/vo_rl
wandb: 🚀 View run at https://wandb.ai/sujit/vo_rl/runs/0mlj4f6f

.........................
Iteration:      3
Total Timestep: 38250
==========================
.....
---------------------------------------------------------------------------------------
^CTraceback (most recent call last):
  File "train.py", line 114, in <module>
    main()
  File "/venv/lib/python3.8/site-packages/hydra/main.py", line 94, in decorated_main
    _run_hydra(
  File "/venv/lib/python3.8/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
    _run_app(
  File "/venv/lib/python3.8/site-packages/hydra/_internal/utils.py", line 457, in _run_app
    run_and_report(
  File "/venv/lib/python3.8/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
    return func()
  File "/venv/lib/python3.8/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
    lambda: hydra.run(
  File "/venv/lib/python3.8/site-packages/hydra/_internal/hydra.py", line 119, in run
    ret = run_job(
  File "/venv/lib/python3.8/site-packages/hydra/core/utils.py", line 186, in run_job
    ret.return_value = task_function(task_cfg)
  File "train.py", line 105, in main
    model.learn(
  File "/home/ruoxi/vo_rl/rl_algorithms/ppo.py", line 347, in learn
    return super().learn(
  File "/home/ruoxi/vo_rl/rl_algorithms/on_policy_algorithm.py", line 287, in learn
    continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
  File "/home/ruoxi/vo_rl/rl_algorithms/on_policy_algorithm.py", line 189, in collect_rollouts
    new_obs, rewards, dones, infos, valid_mask = env.step(clipped_actions, use_gt_initialization=True)
  File "/home/ruoxi/vo_rl/env/svo_wrapper.py", line 128, in step
    images, gt_poses, new_seq = self.get_images_pose()
  File "/home/ruoxi/vo_rl/env/svo_wrapper.py", line 99, in get_images_pose
    batch = next(self.dataloader_iter)
  File "/home/ruoxi/vo_rl/dataloader/tartan_loader.py", line 167, in __getitem__
    images = svo_env.load_image_batch(image_paths, self.num_envs, self.img_h, self.img_w)
KeyboardInterrupt
--------------------------------------------------------------------------------
wandb: | 4.288 MB of 4.288 MB uploaded
wandb: Run history:
wandb:          rollout/iteration ▁▅█
wandb:     rollout/mean_keyframes ▆▁█
wandb: rollout/ratio_valid_stages ▁▃█
wandb:         rollout/sum_reward █▁▅
wandb:   rollout/sum_valid_stages ▁▃█
wandb:            train/approx_kl ██▁
wandb:        train/clip_fraction █▇▁
wandb:           train/clip_range ▁▁▁
wandb:         train/entropy_loss ▁▅█
wandb:   train/explained_variance ▁██
wandb:            train/iteration ▁▅█
wandb:                 train/loss ▁▅█
wandb:            train/n_updates ▁▅█
wandb: train/policy_gradient_loss ▁▅█
wandb:           train/value_loss █▁▁
wandb: 
wandb: Run summary:
wandb:          rollout/iteration 2
wandb:     rollout/mean_keyframes 0.52809
wandb: rollout/ratio_valid_stages 0.19263
wandb:         rollout/sum_reward -1.36541
wandb:   rollout/sum_valid_stages 2456
wandb:            train/approx_kl 0.00372
wandb:        train/clip_fraction 0.07056
wandb:           train/clip_range 0.2
wandb:         train/entropy_loss -2.17906
wandb:   train/explained_variance -6.85537
wandb:            train/iteration 3
wandb:                 train/loss -0.0106
wandb:            train/n_updates 30
wandb: train/policy_gradient_loss -0.00327
wandb:           train/value_loss 0.00113
wandb: 
wandb: 🚀 View run misty-disco-30 at: https://wandb.ai/sujit/vo_rl/runs/eyy4f35w
wandb: ⭐️ View project at: https://wandb.ai/sujit/vo_rl
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./logs/log_voRL/training/experiment_1/Sep07_17-00-13_test/wandb/run-20240907_170022-eyy4f35w/logs
wandb: WARNING The new W&B backend becomes opt-out in version 0.18.0; try it out with `wandb.require("core")`! See https://wandb.me/wandb-core for more information.
```

### Play

- run
```shell
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python play.py 
```

- output
```shell
Matplotlib created a temporary cache directory at /tmp/matplotlib-wq6fuhcj because the default path (/home/ruoxi/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
==========================================================================================
dataset:  tum
dataset_dir:  /home/ruoxi/datasets/TUM-RGBD
params_yaml_path:  /home/ruoxi/vo_rl/svo-lib/svo_env/param/tartan_test.yaml
calib_yaml_path:  /home/ruoxi/vo_rl/svo-lib/svo_env/param/calib/svo_tartanair_pinhole.yaml
num_envs:  1
mode:  val
val_traj_ids:  [0]
==========================================================================================
loaded 1 cameras
  name = cam
  size = [640, 480]
  Projection = Pinhole
  Focal length = (383.013, 382.392)
  Principal point = (344.706, 244.327)
  Distortion: Atan(0.932)
[TUM Dataloader] Loading trajectories
{'rgbd_dataset_freiburg1_360': [100, 756], 'rgbd_dataset_freiburg1_desk': [0, 613], 'rgbd_dataset_freiburg1_desk2': [0, 640], 'rgbd_dataset_freiburg1_floor': [0, 1242], 'rgbd_dataset_freiburg1_plant': [0, 1141], 'rgbd_dataset_freiburg1_room': [0, 1362], 'rgbd_dataset_freiburg1_rpy': [10, 723], 'rgbd_dataset_freiburg1_teddy': [0, 1419], 'rgbd_dataset_freiburg1_xyz': [0, 798]}
self.trajectories_paths:  ['/home/ruoxi/datasets/TUM-RGBD/rgbd_dataset_freiburg1_desk']
self.val_traj_ids:  [0]
len(self.trajectories_paths):  1
self.num_envs:  1
self.trajectories_paths:  ['/home/ruoxi/datasets/TUM-RGBD/rgbd_dataset_freiburg1_desk']
self.val_traj_ids:  [0]
len(self.trajectories_paths):  1
self.num_envs:  1
RMS Loded
100%|███████████████████████████████████████████████████████████████████████████████████████████████▊| 612/613 [00:37<00:00, 16.18it/s]
============ Results ============
Nr Valid States: 609.0
Mean Keyframe  : 1.0
Total Reward   : 1.040047923189999
=================================
```

### evaluate.run

- config param

set the following parameters in `config_eval.yaml`, if you want to `visualize the trajectories` of the evaluation runs.
```yaml
use_rl_actions: True
max_eval_steps: -1  # -1 if maximum length of trajectories should be taken
visualize_trajs: True    
use_gt_initialization: True
```
- run
```shell
ruoxi@robot2go:~/vo_rl$ python -m evaluation.evaluate_runs
```

- output
```shell
Matplotlib created a temporary cache directory at /tmp/matplotlib-p9jvjy4h because the default path (/home/ruoxi/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
Method: with_RL, Number of Keyframes: 250.0
Method: wo_RL, Number of Keyframes: 250.0
Method: with_RL
Number of successful trajectory 1.0/1
Not successful: 
[]
---------------------------------------
Method: wo_RL
Number of successful trajectory 1.0/1
Not successful: 
[]
---------------------------------------
```

### playground

display of `playground` like this gif:

![gif](./img/playground_image_view.gif)

```shell
python3 image_player.py logs/log_voRL/playground/2024-09-14_06-44-47/rgbd_dataset_freiburg1_360 100
```

- arg0 : path to the `playground/xxxxxxxxx/your_test_dataset` folder
- arg1 : `number` of keyframes to play (0~100)
  - 10: quick
  - 30: common
  - 100: slow 