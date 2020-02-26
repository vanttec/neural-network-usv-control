import numpy as np


class Physics():
    '''Physics constants and equations for the quadcopter'''

    def __init__(self, init_eta=None, init_upsilon=None, runtime=None):
        '''Initialize parameters
        Params
        ======
            init_position: 1D numpy array, initial position in x-y-z coordinates (w.r.t. inertial frame)
            init_orientation: 1D numpy array, initial orientation in phi-theta-psi angles (Z-Y-X Tait-Bryan angles)
            init_vel: 1D numpy array, initial velocities in x-y-z components (w.r.t. body frame)
            init_ang_vel: 1D numpy array, initial angle velocities in x-y-z components (w.r.t. body frame)
            runtime: float, limit simulation time in seconds
        '''
        # Initial conditions
        self.init_eta = np.zeros(3) if init_eta is None else np.copy(init_eta)
        self.init_upsilon = np.zeros(3) if init_upsilon is None else np.copy(init_upsilon)

        self.runtime = 5. if runtime is None else runtime

        # Physical constants
        self.integral_step = 1e-4  # Timestep in seconds
        self.X_u_dot = -2.25
        self.Y_v_dot = -23.13
        self.Y_r_dot = -1.31
        self.N_v_dot = -16.41
        self.N_r_dot = -2.79
        self.Xuu = 0
        self.Yvv = -99.99
        self.Yvr = -5.49
        self.Yrv = -5.49
        self.Yrr = -8.8
        self.Nvv = -5.49
        self.Nvr = -8.8
        self.Nrv = -8.8
        self.Nrr = -3.49
        self.m = 30
        self.Iz = 4.1
        self.B = 0.41
        self.c = 0.78

        self.reset()

    def reset(self):
        '''Reset parameters to initial values'''
        self.time = 0.0

        self.eta = np.copy(self.init_eta)
        self.upsilon = np.copy(self.init_upsilon)
        self.eta_dot_last = np.zeros(3)
        self.upsilon_dot_last = np.zeros(3)

        self.done = False

    def reset_random(self, xlim, ylim, yaw_ang_lim, xvel_lim, yvel_lim, yaw_ang_vel_lim):
        '''Reset parameters to random values
        Params
        ======
            xlim: float list, pair of lower and upper limits for x position
            ylim: float list, pair of lower and upper limits for y position
            yaw_ang_lim: float, limit for yaw angle
            vel_lim: float, limit for velocity components
            ang_vel_lim: float, limit for angular velocity components
        '''
        self.time = 0.0

        self.eta = np.random.uniform(*list(zip(xlim, ylim, yaw_ang_lim)))
        self.upsilon = np.random.uniform(*list(zip(xvel_lim, yvel_lim, yaw_ang_vel_lim)))

        self.done = False

    def next_timestep(self, Tport, Tstbd):
        '''Calculate the next state of the quadcopter after one timestep
        Params
        ======
            rotor_speeds: float list, speed of each rotor in rad/s
        Returns
        ======
            bool, end condition reached
        '''
        Xu = -25
        Xuu = 0
        if(abs(self.upsilon[0]) > 1.2):
            Xu = 64.55
            Xuu = -70.92

        Yv = 0.5*(-40*1000*abs(self.upsilon[1])) * \
            (1.1+0.0045*(1.01/0.09) - 0.1*(0.27/0.09)+0.016*(np.power((0.27/0.09), 2)))
        Yr = 6*(-3.141592*1000) * \
            np.sqrt(np.power(self.upsilon[0], 2)+np.power(self.upsilon[1], 2))*0.09*0.09*1.01
        Nv = 0.06*(-3.141592*1000) * \
            np.sqrt(np.power(self.upsilon[0], 2)+np.power(self.upsilon[1], 2))*0.09*0.09*1.01
        Nr = 0.02*(-3.141592*1000) * \
            np.sqrt(np.power(self.upsilon[0], 2)+np.power(self.upsilon[1], 2))*0.09*0.09*1.01*1.01

        M = np.array([[self.m - self.X_u_dot, 0, 0],
                      [0, self.m - self.Y_v_dot, 0 - self.Y_r_dot],
                      [0, 0 - self.N_v_dot, self.Iz - self.N_r_dot]])

        T = np.array([Tport + self.c*Tstbd, 0, 0.5*self.B*(Tport - self.c*Tstbd)])

        CRB = np.array([[0, 0, 0 - self.m * self.upsilon[1]],
                        [0, 0, self.m * self.upsilon[0]],
                        [self.m * self.upsilon[1], 0 - self.m * self.upsilon[0], 0]])

        CA = np.array([[0, 0, 2 * ((self.Y_v_dot*self.upsilon[1]) + ((self.Y_r_dot + self.N_v_dot)/2) * self.upsilon[2])],
                       [0, 0, 0 - self.X_u_dot * self.m * self.upsilon[0]],
                       [2*(((0 - self.Y_v_dot) * self.upsilon[1]) - ((self.Y_r_dot+self.N_v_dot)/2) * self.upsilon[2]), self.X_u_dot * self.m * self.upsilon[0], 0]])

        C = CRB + CA

        Dl = np.array([[0 - Xu, 0, 0],
                       [0, 0 - Yv, 0 - Yr],
                       [0, 0 - Nv, 0 - Nr]])

        Dn = np.array([[Xuu * abs(self.upsilon[0]), 0, 0],
                       [0, self.Yvv * abs(self.upsilon[1]) + self.Yvr * abs(self.upsilon[2]), self.Yrv *
                        abs(self.upsilon[1]) + self.Yrr * abs(self.upsilon[2])],
                       [0, self.Nvv * abs(self.upsilon[1]) + self.Nvr * abs(self.upsilon[2]), self.Nrv * abs(self.upsilon[1]) + self.Nrr * abs(self.upsilon[2])]])

        D = Dl - Dn

        upsilon_dot = np.matmul(np.linalg.inv(
            M), (T - np.matmul(C, self.upsilon) - np.matmul(D, self.upsilon)))
        self.upsilon = (self.integral_step) * (upsilon_dot +
                                               self.upsilon_dot_last)/2 + self.upsilon  # integral
        self.upsilon_dot_last = upsilon_dot

        J = np.array([[np.cos(self.eta[2]), -np.sin(self.eta[2]), 0],
                      [np.sin(self.eta[2]), np.cos(self.eta[2]), 0],
                      [0, 0, 1]])

        eta_dot = np.matmul(J, self.upsilon)  # transformation into local reference frame
        self.eta = (self.integral_step)*(eta_dot+self.eta_dot_last)/2 + self.eta  # integral
        self.eta_dot_last = eta_dot

        if (abs(self.eta[2]) > np.pi):
            self.eta[2] = (self.eta[2]/abs(self.eta[2]))*(abs(self.eta[2])-2*np.pi)

        # End condition
        self.time += self.integral_step
        if self.time >= self.runtime:
            self.done = True
        return self.done
