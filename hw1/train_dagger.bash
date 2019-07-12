#!/bin/bash
set -eux
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
do
    python train_dagger.py --expert-policy-file experts/$e.pkl --envname $e
done