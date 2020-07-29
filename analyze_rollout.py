#!/usr/bin/env python3

import pandas as pd
import numpy as np

tau = np.load("./data/swimmer/swimmer05.npy")[0, :, :4]
theirdata = np.load("./data/swimmer/swimmer05.npy")[0, :, 8:]
ourdata = pd.read_csv("./all_zero_rollout.csv", header=None).values

print(f"theirdata: {theirdata.shape}")
print(f"ourdata: {ourdata.shape}")


def show_timestep(timestep):
    print(f"\nTimestep {timestep} (theirs, ours) (yaw position, yaw vel)")
    print(tau[timestep, :])
    print(np.vstack([theirdata[timestep, :15], ourdata[timestep, :15]]))
    print(np.vstack([theirdata[timestep, 15:], ourdata[timestep, 15:]]))


for i in range(200):
    show_timestep(i)
