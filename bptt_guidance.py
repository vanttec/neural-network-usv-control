from boat import Boat
import numpy as np
import tensorflow.compat.v1 as tf
import math
import matplotlib.pyplot as plt

tf.disable_eager_execution()

def weight_variable(shape):
    '''Create a weight variable for a neural network
    Params
    ======
        shape: int list, shape for the weight variable
    Returns
    ======
        rank-len(shape) tensor, weight variable
    '''
    initial = tf.truncated_normal(shape, stddev=1e-3, dtype=tf.float32)
    return tf.Variable(initial)


class BPTT_Controller():
    '''Implements an Actor Neural Network controller trained with
    Backpropagation Through Time'''

    def __init__(self, boat, train=True, num_hidden_units=[8, 8], batch_size=100, graph_timesteps=20,
                 discount_factor=1.0, train_dt=0.02, train_iterations=1000, model_name=None, graphical=True):
        '''Initialize parameters
        Params
        ======
            quad: Quadcopter object
            train: bool, train the controller
            num_hidden_units: int list, number of hidden units for the neural network (two hidden layers)
            batch_size: int, batch size for parallel computation
            graph_timesteps: int, timesteps for the tensorflow graph
            discount_factor: float, discount factor for total reward calculation
            train_dt: float, timestep for training in seconds
            train_iterations: int, train iterations
            model_name: string, name of the model to restore
            graphical: bool, display graph during training
        '''
        self.boat = boat
        self.train_dt = train_dt
        self.graphical = graphical

        # Neural network parameters
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.action_network_num_inputs = 5
        self.action_network_num_outputs = 2
        self.num_hidden_units = num_hidden_units
                
        if train:
            self.build_graph(graph_timesteps)
            if model_name is not None:
                self.restore_model(model_name)
            self.train(self.random_state, train_iterations)
            self.to_numpy_model()
        else:
            self.reset_random()
            if model_name is not None:
                self.restore_model(model_name)
            else:
                self.sess.run(tf.global_variables_initializer())
            self.to_numpy_model()

    def get_train_target(self):
        '''Use the random position limits of the
        Quadcopter object to generate the train target
        NOTE: the quadcopter is trained in an environment eight times larger
        (x2 for each dimension) in order to cover all posible combinations of
        quadcopter position and target position, as the taget is always set
        at the center of the environment during training
        Returns
        ======
            1D numpy array: train target
        '''
        random_target = []
        for i in range(self.batch_size):
            self.boat.reset_random()
            state = self.boat.state.copy()
            self.boat.reset_random()
            end = self.boat.state.copy()
            train_target = [state[3], state[0], state[1], end[0]*4, end[1]]
            random_target.append(train_target)
        return np.array(random_target)

    def get_random_state(self):
        '''Use the random state generator from the
        Quadcopter object to obtain random states for training
        Size of numpy arrays' first dimension is the batch size for parallel computation
        NOTE: the quadcopter is trained in an environment eight times larger
        (x2 for each dimension) in order to cover all posible combinations of
        quadcopter position and target position, as the taget is always set
        at the center of the environment during training
        Returns
        ======
            2D numpy array: random states for training
        '''
        random_state = []
        random_position = []
        random_ak = []
        for i in range(self.batch_size):
            self.boat.reset_random()
            state = self.boat.state.copy()
            ak = math.atan2(self.train_target[i][4]-self.train_target[i][2],self.train_target[i][3]-self.train_target[i][1])
            ak = np.float32(ak)
            ye = -(state[0] - self.train_target[i][1])*math.sin(ak) + (state[1] - self.train_target[i][2])*math.cos(ak)
            state[3] = 0
            state = np.append(state, ye)
            random_state.append(state[2:7])
            random_position.append(state[0:2])
            random_ak.append(ak)
        return np.array(random_state), np.array(random_position), np.array(random_ak)

    def reset_random(self):
        '''Reset parameters to random values and
        start a new session'''
        tf.reset_default_graph()

        self.W1 = weight_variable([self.action_network_num_inputs, self.num_hidden_units[0]])
        self.b1 = weight_variable([1, self.num_hidden_units[0]])
        self.W2 = weight_variable([self.action_network_num_inputs +
                                   self.num_hidden_units[0], self.num_hidden_units[1]])
        self.b2 = weight_variable([1, self.num_hidden_units[1]])
        self.W3 = weight_variable([self.action_network_num_inputs + self.num_hidden_units[0] +
                                   self.num_hidden_units[1], self.action_network_num_outputs])
        self.b3 = weight_variable([1, self.action_network_num_outputs])

        self.train_target = self.get_train_target()
        self.random_state, self.random_position, self.random_ak = self.get_random_state()

        try:
            self.sess.close()
        except:
            pass
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def run_action_network(self, state):
        '''Run neural network to produce an action
        Size of tensors' first dimension is the batch size for parallel computation
        Params
        ======
            state: rank-2 tensor, state in the form [position,orientation,vel,ang_vel]
        Returns
        ======
            rank-2 tensor, action commands in the form [climb,roll,pitch,yaw]
        '''
        h1 = tf.tanh(tf.matmul(state, self.W1) + self.b1)
        h2 = tf.tanh(tf.matmul(tf.concat([state, h1], axis=1), self.W2) + self.b2)
        action = tf.matmul(tf.concat([state, h1, h2], axis=1), self.W3) + self.b3
        
        action = tf.tanh(action)*np.pi

        return action

    def get_reward(self, ye):
        '''Calculate the reward given a state
        Size of tensors' first dimension is the batch size for parallel computation
        Params
        ======
            state: rank-2 tensor, state in the form [position,orientation,vel,ang_vel]
        Returns
        ======
            rank-1 tensor, reward
        '''
        k_ye = 0.2
        ye = tf.math.abs(ye)
        reward = tf.math.exp(-k_ye*ye)
        return reward

    def next_timestep(self, state, action, position, aux, last):
        '''Calculate the next state of the quadcopter after one timestep
        Size of tensors' first dimension is the batch size for parallel computation
        Params
        ======
            state: rank-2 tensor, state in the form [position,orientation,vel,ang_vel]
            action: rank-2 tensor, action commands in the form [climb,roll,pitch,yaw]
        Returns
        ======
            rank-2 tensor, next state in the form [position,orientation,vel,ang_vel]
        '''
        position = position[:, 0:2]
        psi = tf.reshape(state[:, 0], [self.batch_size, 1])
        eta =  tf.concat([position, psi], 1)
        upsilon = state[:, 1:4]
        ye = tf.reshape(state[:, 4], [self.batch_size, 1])

        psi_d = action[:, 0]

        e_u_int = aux[:, 0]
        Ka_u = aux[:, 1]
        Ka_psi = aux[:, 2]

        eta_dot_last = last[:, 0:3]
        eta_dot_last = tf.reshape(eta_dot_last, [self.batch_size, 3, 1])
        upsilon_dot_last = last[:,3:6]
        upsilon_dot_last = tf.reshape(upsilon_dot_last, [self.batch_size, 3, 1])
        e_u_last = last[:, 6]
        Ka_dot_u_last = last[:, 7]
        Ka_dot_psi_last = last[:, 8]

        zeros = tf.zeros([self.batch_size], dtype=tf.float32)
        ones = tf.ones([self.batch_size], dtype=tf.float32)
        Xu = tf.where(tf.less(tf.abs(upsilon[:, 0]), 1.2), tf.constant(-25, dtype=tf.float32, shape=[
                      self.batch_size]), tf.constant(64.55, dtype=tf.float32, shape=[self.batch_size]))
        Xuu = tf.where(tf.less(tf.abs(upsilon[:, 0]), 1.2), tf.constant(0.0, dtype=tf.float32, shape=[
            self.batch_size]), tf.constant(-70.92, dtype=tf.float32, shape=[self.batch_size]))

        Yv = 0.5*(-40*1000*tf.abs(upsilon[:, 1])) * \
            (1.1+0.0045*(1.01/0.09) - 0.1*(0.27/0.09)+0.016*(tf.pow((0.27/0.09), 2)))
        Yr = 6*(-3.141592*1000) * \
            tf.sqrt(tf.pow(upsilon[:, 0], 2)+tf.pow(upsilon[:, 1], 2))*0.09*0.09*1.01
        Nv = 0.06*(-3.141592*1000) * \
            tf.sqrt(tf.pow(upsilon[:, 0], 2)+tf.pow(upsilon[:, 1], 2))*0.09*0.09*1.01
        Nr = 0.02*(-3.141592*1000) * \
            tf.sqrt(tf.pow(upsilon[:, 0], 2)+tf.pow(upsilon[:, 1], 2))*0.09*0.09*1.01*1.01

        g_u = 1 / (self.boat.physics.m - self.boat.physics.X_u_dot)
        g_psi = 1 / (self.boat.physics.Iz - self.boat.physics.N_r_dot)

        f_u = (((self.boat.physics.m - self.boat.physics.Y_v_dot)*upsilon[:, 1]*upsilon[:, 2] + (Xuu*tf.math.abs(upsilon[:, 0]) + Xu*upsilon[:, 0])) / (self.boat.physics.m - self.boat.physics.X_u_dot))
        f_psi = (((-self.boat.physics.X_u_dot + self.boat.physics.Y_v_dot)*upsilon[:, 0]*upsilon[:, 1] + (Nr * upsilon[:, 2])) / (self.boat.physics.Iz - self.boat.physics.N_r_dot))

        e_u = self.train_target[:, 0] - upsilon[:, 0]

        e_psi = psi_d - eta[:, 2]
        e_psi = tf.where(tf.greater(tf.abs(e_psi), np.pi), (tf.math.sign(e_psi))*(tf.math.abs(e_psi)-2*np.pi), e_psi)

        e_u_int = (self.train_dt)*(e_u + e_u_last)/2 + e_u_int

        e_psi_dot = 0 - upsilon[:, 2]

        sigma_u = e_u + self.boat.physics.lambda_u * e_u_int
        sigma_psi = e_psi_dot + self.boat.physics.lambda_psi * e_psi

        Ka_dot_u = tf.where(tf.greater(Ka_u, self.boat.physics.kmin_u), self.boat.physics.k_u * tf.math.sign(tf.math.abs(sigma_u) - self.boat.physics.mu_u), self.boat.physics.kmin_u*ones)
        Ka_dot_psi = tf.where(tf.greater(Ka_psi, self.boat.physics.kmin_psi), self.boat.physics.k_psi * tf.math.sign(tf.math.abs(sigma_psi) - self.boat.physics.mu_psi), self.boat.physics.kmin_psi*ones)

        Ka_u = self.train_dt*(Ka_dot_u + Ka_dot_u_last)/2 + Ka_u
        Ka_dot_u_last = Ka_dot_u

        Ka_psi = self.train_dt*(Ka_dot_psi + Ka_dot_psi_last)/2 + Ka_psi
        Ka_dot_psi_last = Ka_dot_psi

        ua_u = (-Ka_u * tf.math.sqrt(tf.math.abs(sigma_u)) * tf.math.sign(sigma_u)) - (self.boat.physics.k2_u * sigma_u)
        ua_psi = (-Ka_psi * tf.math.sqrt(tf.math.abs(sigma_psi)) * tf.math.sign(sigma_psi)) - (self.boat.physics.k2_psi * sigma_psi)

        Tx = ((self.boat.physics.lambda_u * e_u) - f_u - ua_u) / g_u
        Tz = ((self.boat.physics.lambda_psi * e_psi) - f_psi - ua_psi) / g_psi

        Tport = (Tx / 2) + (Tz / self.boat.physics.B)
        Tstbd = (Tx / (2*self.boat.physics.c)) - (Tz / (self.boat.physics.B*self.boat.physics.c))

        Tport = tf.where(tf.greater(Tport, 36.5), 36.5*ones, Tport)
        Tport = tf.where(tf.less(Tport, -30), -30*ones, Tport)
        Tstbd = tf.where(tf.greater(Tstbd, 36.5), 36.5*ones, Tstbd)
        Tstbd = tf.where(tf.less(Tstbd, -30), -30*ones, Tstbd)

        M = tf.constant([[self.boat.physics.m - self.boat.physics.X_u_dot, 0, 0],
                         [0, self.boat.physics.m - self.boat.physics.Y_v_dot,
                             0 - self.boat.physics.Y_r_dot],
                         [0, 0 - self.boat.physics.N_v_dot, self.boat.physics.Iz - self.boat.physics.N_r_dot]])

        T = tf.stack([Tport + self.boat.physics.c*Tstbd, zeros, 0.5 *
                      self.boat.physics.B*(Tport - self.boat.physics.c*Tstbd)], axis=1)
        T = tf.reshape(T, [self.batch_size, 3, 1])

        CRB = tf.stack([[zeros, zeros, -self.boat.physics.m * upsilon[:, 1]],
                        [zeros, zeros, self.boat.physics.m * upsilon[:, 0]],
                        [self.boat.physics.m * upsilon[:, 1], -self.boat.physics.m * upsilon[:, 0], zeros]])

        CRB = tf.transpose(CRB, perm=[2, 0, 1])
        
        CA = tf.stack([[zeros, zeros, 2 * ((self.boat.physics.Y_v_dot*upsilon[:, 1]) + ((self.boat.physics.Y_r_dot + self.boat.physics.N_v_dot)/2) * upsilon[:, 2])],
                       [zeros, zeros, -self.boat.physics.X_u_dot * self.boat.physics.m * upsilon[:, 0]],
                       [2*(((-self.boat.physics.Y_v_dot) * upsilon[:, 1]) - ((self.boat.physics.Y_r_dot+self.boat.physics.N_v_dot)/2) * upsilon[:, 2]), self.boat.physics.X_u_dot * self.boat.physics.m * upsilon[:, 0], zeros]])
        CA = tf.transpose(CA, perm=[2, 0, 1])

        C = CRB + CA

        Dl = tf.stack([[-Xu, zeros, zeros],
                       [zeros, -Yv, -Yr],
                       [zeros, -Nv, -Nr]])
        Dl = tf.transpose(Dl, perm=[2, 0, 1])

        Dn = tf.stack([[Xuu * tf.abs(upsilon[:, 0]), zeros, zeros],
                       [zeros, self.boat.physics.Yvv * tf.abs(upsilon[:, 1]) + self.boat.physics.Yvr * tf.abs(upsilon[:, 2]), self.boat.physics.Yrv *
                           tf.abs(upsilon[:, 1]) + self.boat.physics.Yrr * tf.abs(upsilon[:, 2])],
                       [zeros, self.boat.physics.Nvv * tf.abs(upsilon[:, 1]) + self.boat.physics.Nvr * tf.abs(upsilon[:, 2]), self.boat.physics.Nrv * tf.abs(upsilon[:, 1]) + self.boat.physics.Nrr * tf.abs(upsilon[:, 2])]])
        Dn = tf.transpose(Dn, perm=[2, 0, 1])

        D = Dl - Dn

        upsilon = tf.reshape(upsilon, [self.batch_size, 3, 1])
        upsilon_dot = tf.matmul(tf.linalg.inv(
            M), (T - tf.matmul(C, upsilon) - tf.matmul(D, upsilon)))

        upsilon = (self.train_dt) * (upsilon_dot + upsilon_dot_last)/2 + upsilon  # integral

        J = tf.stack([[tf.cos(eta[:, 2]), -tf.sin(eta[:, 2]), zeros],
                      [tf.sin(eta[:, 2]), tf.cos(eta[:, 2]), zeros],
                      [zeros, zeros, ones]])
        J = tf.transpose(J, perm=[2, 0, 1])

        eta_dot = tf.matmul(J, upsilon)  # transformation into local reference frame
        eta = tf.reshape(eta, [self.batch_size, 3, 1])
        eta = (self.train_dt) * (eta_dot + eta_dot_last)/2 + eta  # integral

        psi = eta[:, 2]
        psi = tf.where(tf.greater(tf.abs(psi), np.pi), (tf.math.sign(psi))*(tf.math.abs(psi)-2*np.pi), psi)

        ye = -(position[:, 0] - self.train_target[:, 1])*tf.math.sin(self.random_ak) + (position[:, 1] - self.train_target[:, 2])*tf.math.cos(self.random_ak)

        psi = tf.reshape(psi, [self.batch_size, 1])
        upsilon = tf.reshape(upsilon, [self.batch_size, 3])
        ye = tf.reshape(ye, [self.batch_size, 1])
        next_state = tf.concat([psi, upsilon, ye], 1)

        eta = tf.reshape(eta, [self.batch_size, 3])
        next_position = tf.reshape(eta[:, 0:2], [self.batch_size, 2])

        e_u_int = tf.reshape(e_u_int, [self.batch_size, 1])
        Ka_u = tf.reshape(Ka_u, [self.batch_size, 1])
        Ka_psi = tf.reshape(Ka_psi, [self.batch_size, 1])
        next_aux = tf.concat([e_u_int, Ka_u, Ka_psi], 1)

        eta_dot = tf.reshape(eta_dot, [self.batch_size, 3])
        upsilon_dot = tf.reshape(upsilon_dot, [self.batch_size, 3])
        e_u = tf.reshape(e_u, [self.batch_size, 1])
        Ka_dot_u = tf.reshape(Ka_dot_u, [self.batch_size, 1])
        Ka_dot_psi = tf.reshape(Ka_dot_psi, [self.batch_size, 1])
        next_last = tf.concat([eta_dot, upsilon_dot, e_u, Ka_dot_u, Ka_dot_psi], 1)

        reward = self.get_reward(ye)

        return next_state, reward, next_position, next_aux, next_last

    def build_graph(self, graph_timesteps):
        '''Build a tensorflow graph to train using an optimizer
        Params
        ======
            graph_timesteps: int, timesteps for the tensorflow graph
        '''

        self.reset_random()
        self.graph_timesteps = graph_timesteps
        total_reward = tf.zeros([self.batch_size], dtype=tf.float32)

        # Placeholder
        self.ph_initial_state = tf.placeholder(
            tf.float32, shape=[self.batch_size, self.action_network_num_inputs])
        state = self.ph_initial_state

        self.ph_initial_position = tf.placeholder(tf.float32, shape=[self.batch_size, 2])
        position = self.ph_initial_position

        self.ph_initial_aux = tf.placeholder(tf.float32, shape=[self.batch_size, 3])
        aux = self.ph_initial_aux

        self.ph_initial_lasts = tf.placeholder(tf.float32, shape=[self.batch_size, 9])
        last = self.ph_initial_lasts

        # Unrolled trajectory graph
        actions = []
        rewards = []
        trajectory = [state]
        positions = [position]
        auxs = [aux]
        lasts = [last]
        for t in range(self.graph_timesteps):
            action = self.run_action_network(state)
            state, reward, position, aux, last = self.next_timestep(state, action, position, aux, last)
            total_reward += (self.discount_factor ** t) * reward
            actions.append(action)
            rewards.append(reward)
            trajectory.append(state)
            positions.append(position)
            auxs.append(aux)
            lasts.append(last)

        self.actions = tf.stack(actions)
        self.rewards = tf.stack(rewards)
        self.trajectory = tf.stack(trajectory)
        self.positions = tf.stack(positions)
        self.auxs = tf.stack(auxs)
        self.lasts = tf.stack(lasts)

        # Initialize optimizer
        self.average_total_reward = tf.reduce_mean(total_reward)
        optimizer = tf.train.AdamOptimizer()
        self.update = optimizer.minimize(-self.average_total_reward)
        self.sess.run(tf.global_variables_initializer())

    def train(self, initial_state, train_iterations):
        '''Train the neural network
        Size of tensors' first dimension is the batch size for parallel computation
        Params
        ======
            initial_state: rank-2 tensor, initial state in the form [position,orientation,vel,ang_vel]
            train_iterations: int, train iterations
        '''
        # Initialize plot
        if self.graphical:
            plt.close()
            fig = plt.figure('Trajectory')
            ylabels = ['psi', 'u', 'v', 'r', 'ye',
                       'Tport', 'Tstbd', 'avg. total reward']
            axes = []
            for i in range(11):
                ax = fig.add_subplot(3, 3, i + 1)
                ax.xaxis.set_label_coords(1.05, -0.025)
                plt.xlabel('time')
                plt.ylabel(ylabels[i])
                for j in range(self.batch_size):
                    ax.plot([])
                axes.append(ax)
            plt.xlabel('iterations')

        initial_position = np.zeros([self.batch_size, 2])
        initial_auxs = np.zeros([self.batch_size, 3])
        initial_lasts = np.zeros([self.batch_size, 9])

        # Train
        iterations = []
        average_total_rewards = []
        for i in range(train_iterations + 1):            
            # Get results

            initial_state = self.random_state

            actions, rewards, trajectory, average_total_reward = self.sess.run(
                [self.actions, self.rewards, self.trajectory, self.average_total_reward],
                feed_dict={self.ph_initial_state: initial_state, self.ph_initial_position: initial_position, 
                self.ph_initial_aux: initial_auxs, self.ph_initial_lasts: initial_lasts})

            # Update weight variables
            if i != train_iterations:
                self.sess.run(self.update, feed_dict={self.ph_initial_state: initial_state, self.ph_initial_position: initial_position, self.ph_initial_aux: initial_auxs, self.ph_initial_lasts: initial_lasts})

            # Plot results
            iterations.append(i)
            average_total_rewards.append(average_total_reward)
            print("iteration: ", i, 'Average total reward:', average_total_reward)
            if i % 50 == 0:
                print('iteration:', i, 'Average total reward:', average_total_reward)
                print("train target: ", self.train_target[:,1])
                if self.graphical:
                    for traj in range(self.batch_size):
                        time = np.arange(self.graph_timesteps + 1) * self.train_dt
                        for j in range(5):
                            del axes[j].lines[0]
                            axes[j].plot(time, trajectory[:, traj, j], 'b-')
                            axes[j].relim()
                        for j in range(2):
                            del axes[j+5].lines[0]
                            axes[j+5].plot(time[:-1], actions[:, traj, j], 'b-')
                            axes[j+5].relim()
                    del axes[7].lines[0]
                    axes[7].plot(iterations, average_total_rewards, 'b-')
                    plt.pause(1e-10)
                

            if (i > 1 and i % 1000 == 0):
                #self.train_target = self.get_train_target()
                self.random_state = self.get_random_state()
                #initial_state = self.random_state
                self.save_model('example'+ str(i)) #find a way to  


    def save_model(self, model_name):
        '''Save weight variables
        Params
        ======
            model_name: string, name of the model to save
        '''
        self.saver.save(self.sess, './models/' + model_name,
                        write_meta_graph=False, write_state=False)

    def restore_model(self, model_name):
        '''Restore weight variables
        Params
        ======
            model_name: string, name of the model to restore
        '''
        self.saver.restore(self.sess, './models/' + model_name)

    def to_numpy_model(self):
        '''Convert tensorflow weight variables into numpy weight variables'''
        self.np_W1 = self.sess.run(self.W1)
        self.np_b1, = self.sess.run(self.b1)
        self.np_W2 = self.sess.run(self.W2)
        self.np_b2, = self.sess.run(self.b2)
        self.np_W3 = self.sess.run(self.W3)
        self.np_b3, = self.sess.run(self.b3)

    def run_numpy_action_network(self, state):
        '''Run neural network to produce an action
        Params
        ======
            state: 1D numpy array, state in the form [position,orientation,vel,ang_vel]
        Returns
        ======
            1D numpy array, action commands in the form [climb,roll,pitch,yaw]
        '''
        h1 = np.tanh(np.matmul(state, self.np_W1) + self.np_b1)
        h2 = np.tanh(np.matmul(np.concatenate([state, h1]), self.np_W2) + self.np_b2)
        action = np.matmul(np.concatenate([state, h1, h2]), self.np_W3) + self.np_b3
        return action

    def position_control(self, target):
        '''Calculate rotor speeds to stabilize
        or to reach a target if preset
        Params
        ======
            target: 1D numpy array, target position in x-y-z coordinates
        Returns
        ======
            float list, speed for each rotor in rad/s
        '''
        state = self.boat.state.copy()
        print(state)
        state[0:3] -= (target - self.train_target)
        action = self.run_numpy_action_network(state)
        rotor_speeds = to_rotor_speeds(self.boat.physics.hover_rotor_speed, *action)
        return rotor_speeds

    def reset(self):
        '''Used in simulation.py to reset the parameters of the contrroler
        between simulations (e.g. initial error and integral in a PID controller)
        It is not necessary to implement for a trained neural network'''
        pass


'''To train'''

# Eta limits
xlim = [-5, 5]
ylim = [-5, 5]
yawlim = [-np.pi, np.pi]
# Random upsilon limits
xvel_lim = [0.4, 1.2]
yvel_lim = [-0.0, 0.0]
yaw_ang_vel_lim = [-0.0, 0.0]
# Lists of parameters
eta_limits = [xlim, ylim, yawlim]
upsilon_limits = [xvel_lim, yvel_lim, yaw_ang_vel_lim]
# Create objects
#total_iterations=50000
train_iterations= 50000
#cycles = total_iterations/train_iterations
#for k in range(int(cycles)):
#    n=(k)*train_iterations
#    if k==0:
model_name= None#'example'+ str(9000)
#    else: 
#        model_name='example'+ str(n)
    # Create objects
boat = Boat(random_eta_limits=eta_limits, random_upsilon_limits=upsilon_limits)
ctrl = BPTT_Controller(boat, train=True, num_hidden_units=[64, 64], graph_timesteps=250, train_dt=0.01, train_iterations=train_iterations, model_name=model_name)
ctrl.save_model('example'+ str(train_iterations))
#    del ctrl

print(total_iterations)
