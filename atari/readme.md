# Atari DQN

This is an implementation of the DQN approach applied to an Atari game. The game is loaded through openAI gym.
This script can be used to train the DQN but also to play without training to look at results.

Using the DQN algorithm described here https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

The checkpoints of this trained network are all listed in the checkpoint directory. In the section *results* an overview of the result of every checkpoint is presented, with tensorboard screenshots and video renders.

## Usage
To use the script, simply run
> python atari.py

All the possible running arguments can be seen running
>python atari.py --help

## Results
In these results we will present mean episode rewards for the different checkpoints. Results are presented both in vanilla mode (no reward clipping, no episode end on life loss) and in DQN mode (reward clipping and episode is 1 life).

I trained the network for 24h+ on a Tesla K80 GPU, resulting in ~8M steps (or ~50k episodes). These are the relative tensorboard screenshots and relative meaning (in the scale of number of episodes):

![tensorboard1](src/screen1.png?raw=true)
![tensorboard2](src/screen2.png?raw=true)
![tensorboard3](src/screen3.png?raw=true)

- **Episode steps**: number of steps for every episode
- **Epsilon**: decay of the epsilon-greedy parameter (linear in the number of steps, not in the number of episodes)
- **Mean_max_q**: episode-mean of the max of the compute Q-value by the DQN network
- **Reward**: episode reward
- **Reward100**: mean reward for the last 100 episodes
- **Reward1000**: mean reward for the last 1000 episodes

### Testing
I tested the checkpoints both in dqn and vanilla mode, here the mean of 100 episodes is reported.

| Steps | DQN mode | Vanilla mode | Render       |
| ----- | -------- | ------------ | -------------|
| 200k  | 0.96     | 6.42         | ![200k][200k]|
| 520k  | 2.63     | 12.99        | ![520k][520k]|
| 770k  | 2.8      | 18.4         | ![770k][770k]|
| 960k  | 2.88     | 15.85        | ![960k][960k]|
| 1.8M  | 4.28     | 28.03        | ![1_8M][18M]|
| 4.8M  | 5.84     | 46.01        | ![4_8M][48M]|
| 6.2M  | 7.13     | 65.08        | ![6_2M][62M]|
| 7.9M  | 6.88     | 57.61        | ![7_9M][79M]|

[200k]: src/200k.gif?raw=true
[520k]: src/520k.gif?raw=true
[770k]: src/770k.gif?raw=true
[960k]: src/960k.gif?raw=true
[18M]: src/18M.gif?raw=true
[48M]: src/48M.gif?raw=true
[62M]: src/62M.gif?raw=true
[79M]: src/79M.gif?raw=true
