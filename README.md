# Deep Learning — Coursework & Projects (MSAI)

This repository contains coursework and projects from the Deep Learning class. The work focuses on modern deep architectures (MLP, CNN, Transformer) and practical applications using images captured from the SuperTuxKart simulator.

## What I learned

- Convolutional Neural Networks (CNNs)
- Transformer architectures and self-attention
- Attention mechanisms and positional encodings
- Pooling, convolutions, and feature extraction
- Training deep networks end-to-end for perception tasks

## Projects and experiments

1. Training deep networks and MLPs to classify images collected from SuperTuxKart. Models include simple MLP baselines and deeper convolutional models.

2. Building Convolutional Networks to perform core vision tasks on SuperTuxKart imagery:

   - Classification (what is in the frame)
   - Segmentation (which pixels belong to road, obstacles, etc.)
   - Detection (bounding boxes for objects)

3. Final autonomous driving project: combining CNNs, Transformers, and MLP modules to predict controls and make the kart drive autonomously inside the simulator.

## Dataset

All images and data used in the experiments were collected from SuperTuxKart. See the homework folders for the raw data and any preprocessing code.

## Final project videos

The recorded runs for the final project are in `Homework 4/videos/`.

Below are inline players for the three recorded planners. If your viewer doesn't render HTML5 video, the links below each player will open the files directly.

<video controls width="480" poster="Homework 4/videos/cnn_planner_lighthouse.mp4">
   <source src="Homework 4/videos/cnn_planner_lighthouse.mp4" type="video/mp4">
   Your browser does not support the video tag. Download: <a href="Homework 4/videos/cnn_planner_lighthouse.mp4">CNN planner</a>
</video>

**CNN planner** — [Open file](Homework 4/videos/cnn_planner_lighthouse.mp4)

<video controls width="480" poster="Homework 4/videos/mlp_planner_lighthouse.mp4">
   <source src="Homework 4/videos/mlp_planner_lighthouse.mp4" type="video/mp4">
   Your browser does not support the video tag. Download: <a href="Homework 4/videos/mlp_planner_lighthouse.mp4">MLP planner</a>
</video>

**MLP planner** — [Open file](Homework 4/videos/mlp_planner_lighthouse.mp4)

<video controls width="480" poster="Homework 4/videos/transformer_planner_lighthouse.mp4">
   <source src="Homework 4/videos/transformer_planner_lighthouse.mp4" type="video/mp4">
   Your browser does not support the video tag. Download: <a href="Homework 4/videos/transformer_planner_lighthouse.mp4">Transformer planner</a>
</video>

**Transformer planner** — [Open file](Homework 4/videos/transformer_planner_lighthouse.mp4)

## Results & notes

The final project demonstrates autonomous control using three modeling approaches (CNN, MLP, Transformer). Video links above show recorded runs on the Lighthouse track.

The CNN planner uses a convolutional architecture to process the input images and predict steering commands. The MLP planner applies a multi-layer perceptron to the same task, while the Transformer planner leverages self-attention mechanisms for decision-making.

The results show that all three planners can navigate the track, with varying degrees of success. The CNN planner generally performs well, effectively recognizing road features and obstacles. The MLP planner is simpler but still manages to drive reasonably well. The Transformer planner, while more complex, shows promise in handling longer-term dependencies in the driving task.
