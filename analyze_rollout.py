#!/usr/bin/env python3

import pandas as pd
import numpy as np

theirdata = np.load("./data/swimmer/swimmer05.npy")[0, :, 8:]
ourdata = pd.read_csv("./all_zero_rollout.csv", header=None).values

print(f"theirdata: {theirdata.shape}")
print(f"ourdata: {ourdata.shape}")


def show_timestep(timestep):
    print(f"\nTimestep {timestep} (theirs, ours) (yaw position, yaw vel)")
    print(np.vstack([theirdata[timestep, 2:15:3], ourdata[timestep, 2:15:3]]))
    print(np.vstack([theirdata[timestep, 17:30:3], ourdata[timestep,
                                                           17:30:3]]))


for i in range(5):
    show_timestep(i)
