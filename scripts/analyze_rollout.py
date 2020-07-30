#!/usr/bin/env python3

import pandas as pd
import numpy as np

tau = np.load("./data/swimmer/swimmer05.npy")[0, :, :4]
mjcdata = np.load("./data/swimmer/swimmer05.npy")[0, :, 8:]
tdsdata = pd.read_csv("./all_zero_rollout.csv", header=None).values
pybdata = pd.read_csv("./pybullet_rollout.csv", header=None).values

print(f"mjcdata: {mjcdata.shape}")
print(f"tdsdata: {tdsdata.shape}")
print(f"pybdata: {tdsdata.shape}")


def yaws(a):
    return a.reshape(-1, 3)[:, 2]


def show_timestep(timestep):
    print(f"\nTimestep {timestep}")
    print("tau:", tau[timestep, :])
    mjcpos = mjcdata[timestep, :15]
    tdspos = tdsdata[timestep, :15]
    pybpos = pybdata[timestep, :15]
    print("mjc pos:", yaws(mjcpos))
    print("tds pos:", yaws(tdspos))
    print("pyb pos:", yaws(pybpos))

    if (tdsdata.shape[1] == 30):
        mjcvel = mjcdata[timestep, 15:]
        tdsvel = tdsdata[timestep, 15:]
        pybvel = pybdata[timestep, 15:]
        print("mjc vel:", yaws(mjcvel))
        print("tds vel:", yaws(tdsvel))
        print("pyb vel:", yaws(pybvel))


for i in range(200):
    show_timestep(i)
