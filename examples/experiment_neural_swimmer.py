import pybullet as p
import time
import numpy as np

NLINKS = 5
NJOINTS = NLINKS - 1


def main():
    data = np.load(f"data/swimmer{NLINKS:02d}.npy")
    _, totalsteps, _ = data.shape

    p.connect(p.GUI)
    swimmer = p.loadURDF(
        f"data/swimmer/swimmer{NLINKS:02d}/swimmer{NLINKS:02d}.urdf",
        flags=p.URDF_USE_INERTIA_FROM_FILE)

    # Turn off zero-velocity holding motors.
    for joint in range(p.getNumJoints(swimmer)):
        p.setJointMotorControl2(swimmer, joint, p.VELOCITY_CONTROL, force=0)

    dt = 1 / 500
    p.setTimeStep(dt)  # From mjcf
    traj = data[0, :, :]
    torques = traj[:, :NJOINTS]

    for timestep in range(totalsteps):
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

        p.stepSimulation()

        joint_states = p.getJointStates(swimmer,
                                        range(p.getNumJoints(swimmer)))
        q = [s[0] for s in joint_states]
        qd = [s[1] for s in joint_states]

        print(f"{timestep:04d} tau: {applied_torques}")
        print(f"{timestep:04d}   q: {q}")
        print(f"{timestep:04d}  qd: {qd}")

        time.sleep(100 * dt)


if __name__ == '__main__':
    main()
