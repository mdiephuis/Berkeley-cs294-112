#!/bin/bash
set -eux
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
do
    python train_behavior_clone.py --envname $e
done