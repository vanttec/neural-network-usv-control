import numpy as np
from graph3d import Graph3D
from boat import Boat
from bptt import BPTT_Controller
import matplotlib.pyplot as plt

# Simulation setup
runtime = 10.
num_simulations = 3

# Display vectors
traj = True
vel = True
ang_vel = False
acc = False
ang_acc = False
force = True
torque = False
rotor = True

# Graph3D limits
xlim = [-2.5, 2.5]
ylim = [-2.5, 2.5]
zlim = [0.0, 5.0]

# Random state generator limits
yaw_ang_lim = 1. * np.pi  # rad
vel_lim = 1.  # m/s
ang_vel_lim = 1. * np.pi  # rad/s

# Lists of parameters
display_vectors = [traj, vel, ang_vel, acc, ang_acc, force, torque, rotor]
graph_limits = [xlim, ylim, zlim]
random_state_limits = [yaw_ang_lim, vel_lim, ang_vel_lim]

# Create objects
boat = Boat(runtime=runtime,
            random_position_limits=graph_limits[0:2],
            random_state_limits=random_state_limits,
            results_interval=20)
boat.physics.dt = 0.001
ctrl = BPTT_Controller(boat, model_name='target_v2', num_hidden_units=[64, 64])
graph3d = Graph3D(*graph_limits)

# Run simulation
data = []
targets = []
results = []
np.random.seed(1)
for i in range(num_simulations):
    target = np.random.uniform(*list(zip(*graph_limits[0:2])))
    boat.reset_random()
    ctrl.reset()
    while True:
        rotor_speeds = ctrl.position_control(target)
        _, done = boat.step(rotor_speeds)
        if done:
            break
    dat = boat.get_graph3d_data(target, *display_vectors)
    data.append(dat)
    targets.append(target)
    results.append(boat.results)

# Plot Graph3D results
for dat in data:
    graph3d.show(*dat)

# Plot specific results


def plot_results(idx):
    plt.figure(2)
    plt.clf()
    plt.plot(results[idx]['time'], results[idx]['x'], label='x')
    plt.plot(results[idx]['time'], results[idx]['y'], label='y')
    plt.plot(results[idx]['time'], results[idx]['z'], label='z')
    plt.legend()
    plt.pause(0.1)

    plt.figure(3)
    plt.clf()
    plt.plot(results[idx]['time'], results[idx]['phi'], label='roll angle')
    plt.plot(results[idx]['time'], results[idx]['theta'], label='pitch angle')
    plt.plot(results[idx]['time'], results[idx]['psi'], label='yaw angle')
    plt.legend()
    plt.pause(0.1)

    plt.figure(4)
    plt.clf()
    plt.plot(results[idx]['time'], results[idx]['rotor_speed1'], label='Rotor 1 rad/s')
    plt.plot(results[idx]['time'], results[idx]['rotor_speed2'], label='Rotor 2 rad/s')
    plt.plot(results[idx]['time'], results[idx]['rotor_speed3'], label='Rotor 3 rad/s')
    plt.plot(results[idx]['time'], results[idx]['rotor_speed4'], label='Rotor 4 rad/s')
    plt.legend()
    plt.pause(0.1)
