# OpenArm Folding

A simulated OpenArm robot learns to throw a colored ball into the matching colored floor bin.

Give it an instruction such as:

> "Throw the red ball into the blue bucket."

The right arm must identify the correct ball, pick it up from the side table, and throw it into the matching bin. The left arm remains stationary. The scene contains five balls and five bins, with their colors shuffled between episodes, so the ball and target bin are not necessarily the same color.

## Current Setup

A scripted expert currently performs the throwing task. The same simulation is configured to record expert demonstrations for training and fine-tuning a vision-language-action (VLA) model, [SmolVLA](https://huggingface.co/blog/smolvla), which will control the arm using camera observations and the natural-language instruction.

## Expert Demonstration

![Successful expert throw](artifacts/readme%20artifacts/Wrong%20Bucket.gif)
