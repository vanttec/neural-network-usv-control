import numpy as np
from physics import Physics


class Boat():
    '''Boat environment'''

    def __init__(self, init_eta=None, init_upsilon=None, runtime=None, results_interval=300,
                 random_eta_limits=None, random_upsilon_limits=None):
        '''Initialize parameters
        Params
        ======
            init_position: 1D numpy array, initial position in x-y-z coordinates (w.r.t. inertial frame)
            init_orientation: 1D numpy array, initial orientation in phi-theta-psi angles (Z-Y-X Tait-Bryan angles)
            init_vel: 1D numpy array, initial velocities in x-y-z components (w.r.t. body frame)
            init_ang_vel: 1D numpy array, initial angle velocities in x-y-z components (w.r.t. body frame)
            runtime: float, limit simulation time in seconds
            results_interval: int, iterations between saving results
            random_position_limits: float list, limits of random position generator
            random_state_limits: float list, limits of random state generator (orientation, vel and ang_vel)
        '''
        self.labels = ['time', 'x', 'y', 'psi',
                       'x_velocity', 'y_velocity',
                       'z_angular_velocity',
                       'Tport', 'Tstbd']
        self.physics = Physics(init_eta, init_upsilon, runtime)
        self.results_interval = results_interval
        self.random_eta_limits = [[-2.5, 2.5],
                                  [-2.5, 2.5],
                                  [-np.pi, np.pi]] if random_eta_limits is None else random_eta_limits
        self.random_upsilon_limits = [[0.0, 0.0],
                                      [0.0, 0.0],
                                      [0.0, 0.0]] if random_upsilon_limits is None else random_upsilon_limits
        self.reset()

    def reset(self):
        '''Reset environment to initial values'''
        self.iteration_count = 0
        self.results = {x: [] for x in self.labels}
        self.physics.reset()
        self.state = self._get_state()

    def reset_random(self):
        '''Reset environment to random values'''
        self.iteration_count = 0
        self.results = {x: [] for x in self.labels}
        self.physics.reset_random(*self.random_eta_limits,
                                  *self.random_upsilon_limits)
        self.state = self._get_state()

    def _get_state(self):
        '''Get the current vector state
        Returns
        ======
            1D numpy array, state vector in the form [position,orientation,vel,ang_vel]
        '''
        state = np.concatenate([self.physics.eta, self.physics.upsilon])
        return state

    def set_state(self, state):
        '''Set current vector state
        Params
        ======
            state: 1D numpy array, state vector in the form [position,orientation,vel,ang_vel]
        '''
        self.state = np.copy(state)
        self.physics.eta = self.state[0:3]
        self.physics.upsilon = self.state[3:6]

    def step(self, Tport, Tstbd):
        '''Use the rotor speeds to obtain the next state
        Params
        ======
            rotor_speeds: float list, speed of each rotor in rad/s
        Returns
        ======
            1D numpy array, next state in the form [position,orientation,vel,ang_vel]
            bool, end condition reached
        '''
        done = self.physics.next_timestep(Tport, Tstbd)
        self.state = self._get_state()

        # Save results
        if self.iteration_count % self.results_interval == 0:
            results = [self.physics.time] + list(self.physics.eta)
            results += list(self.physics.upsilon)
            results += [Tport, Tstbd]
            for i in range(len(self.labels)):
                self.results[self.labels[i]].append(results[i])
        self.iteration_count += 1
        return self.state, done
