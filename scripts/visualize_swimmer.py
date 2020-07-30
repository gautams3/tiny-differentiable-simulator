import pybullet as p
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

NLINKS = 5
NJOINTS = NLINKS - 1

argparser = argparse.ArgumentParser()
argparser.add_argument("--pybullet_physics", action="store_true")
argparser.add_argument("--slowdown", type=int, default=100)
args = argparser.parse_args()


def visualize_tds():
    p.connect(p.GUI)
    p.setRealTimeSimulation(False)
    swimmer = p.loadURDF(
        f"data/swimmer/swimmer{NLINKS:02d}/swimmer{NLINKS:02d}.urdf",
        flags=p.URDF_USE_INERTIA_FROM_FILE)

    qs = pd.read_csv("./all_zero_qs.csv", header=None).values
    timesteps, _ = qs.shape

    dt = 1 / 500

    for t in tqdm(range(timesteps)):
        for j, q in enumerate(qs[t, :]):
            p.resetJointState(swimmer, j, q)
        p.stepSimulation()
        time.sleep(args.slowdown * dt)

    p.disconnect()


def pybullet_physics():
    data = np.load(f"data/swimmer/swimmer{NLINKS:02d}.npy")
    _, totalsteps, _ = data.shape

    p.connect(p.GUI)
    p.setRealTimeSimulation(False)
    swimmer = p.loadURDF(
        f"data/swimmer/swimmer{NLINKS:02d}/swimmer{NLINKS:02d}.urdf",
        flags=p.URDF_USE_INERTIA_FROM_FILE)

    # Turn off zero-velocity holding motors.
    for joint in range(p.getNumJoints(swimmer)):
        p.setJointMotorControl2(swimmer, joint, p.VELOCITY_CONTROL, force=0)

    traj = data[0, :, :]
    torques = traj[:, :NJOINTS]
    q0 = traj[0, NJOINTS:2 * NJOINTS]
    for j, q in enumerate(q0):
        p.resetJointState(swimmer, j + 3, q)
    head_yaw = traj[0, 2 * NJOINTS + 2]
    p.resetJointState(swimmer, 2, head_yaw)

    dt = 1 / 500
    p.setTimeStep(dt)  # From mjcf
    control_steps = 10  # per env

    with open("pybullet_rollout.csv", "w") as f:
        for timestep in tqdm(range(totalsteps)):
            applied_torques = []
            for joint in range(NJOINTS):
                urdfjoint = joint + 3  # x + y + yaw
                torque = torques[timestep, joint]
                applied_torques.append(torque)
                print(urdfjoint, p.getNumJoints(swimmer))
                p.setJointMotorControl2(swimmer,
                                        urdfjoint,
                                        p.TORQUE_CONTROL,
                                        force=torque)
                dynamics_info = p.getDynamicsInfo(swimmer, urdfjoint)
                print(dynamics_info[2], dynamics_info[3], dynamics_info[4])

            state = []
            for link in range(2, 2 + NLINKS):
                link_state = p.getLinkState(swimmer, link)
                state.append(link_state[0][0])
                state.append(link_state[0][1])
                mat = p.getMatrixFromQuaternion(link_state[1])
                state.append(np.arctan2(-mat[0], mat[1]))
            f.write(','.join(map(str, state)) + "\n")

            for _ in range(control_steps):
                p.stepSimulation()

            joint_states = p.getJointStates(swimmer,
                                            range(p.getNumJoints(swimmer)))
            q = [s[0] for s in joint_states]
            qd = [s[1] for s in joint_states]

            print(f"{timestep:04d} tau: {applied_torques}")
            print(f"{timestep:04d}   q: {q}")
            print(f"{timestep:04d}  qd: {qd}")

            time.sleep(args.slowdown / control_steps * dt)

    p.disconnect()


if __name__ == '__main__':
    if args.pybullet_physics:
        pybullet_physics()
    else:
        visualize_tds()
