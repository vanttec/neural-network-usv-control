from boat import Boat
import numpy as np
import tensorflow.compat.v1 as tf
#import tensorflow as tf2
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
        self.action_network_num_inputs = 7
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
        self.boat.reset_random()
        state = self.boat.state.copy()
        train_target = [0,state[3]]
        return np.array(train_target)

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
        for i in range(self.batch_size):
            self.boat.reset_random()
            state = self.boat.state.copy()
            if self.train_target[1] > 1.2:
                state[2] = state[2]*0.1
            elif self.train_target[1] > 0.7:
                state[2] = state[2]*0.5
            e_u = self.train_target[1] - state[3]
            state = np.append(state, e_u)
            state = np.append(state, 0)
            state = np.append(state, 0)
            '''for j in range(3):
                state[j] *= 2.'''
            random_state.append(state[2:9])
        return np.array(random_state)

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
        self.random_state = self.get_random_state()

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
        
        action = tf.tanh(action)*35

        return action

    def get_reward(self, state, port, stbd):
        '''Calculate the reward given a state
        Size of tensors' first dimension is the batch size for parallel computation
        Params
        ======
            state: rank-2 tensor, state in the form [position,orientation,vel,ang_vel]
        Returns
        ======
            rank-1 tensor, reward
        '''
        ku = 3
        kpsi = 5.72
        kt = 3

        heading = state[:,0]
        e_u = tf.math.abs(state[:,4])
        e_u = tf.reshape(e_u, [self.batch_size, 1])
        e_psi = tf.math.abs(self.train_target[0] - heading)
        e_psi = tf.reshape(e_psi, [self.batch_size, 1])
        port = tf.reshape(port, [self.batch_size, 1])
        stbd = tf.reshape(stbd, [self.batch_size, 1])
        reward = 0.35 * tf.math.exp(-ku*e_u)
        reward_psi = tf.where(tf.less(e_psi, np.pi/2), tf.math.exp(-kpsi*e_psi), tf.tanh(-tf.math.exp(kpsi*(e_psi-np.pi))))
        reward += 0.45 * reward_psi
        reward += 0.1 * tf.where(tf.less(port, 3.5), tf.tanh(tf.math.exp(-kt*port)), tf.tanh(-tf.math.exp(kt*(port-6))))
        reward += 0.1 * tf.where(tf.less(stbd, 3.5), tf.tanh(tf.math.exp(-kt*stbd)), tf.tanh(-tf.math.exp(kt*(stbd-6))))
        return reward, e_psi

    def next_timestep(self, state, action, port, stbd, last):
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
        eta = state[:, 0]
        upsilon = state[:, 1:4]
        Tport = action[:, 0]
        Tstbd = action[:, 1]
        #past_port = state[:, 5]
        #past_stbd = state[:, 6]
        eta_dot_last = last[:, 0]
        upsilon_dot_last = last[:,1:4]
        eta_dot_last = tf.reshape(eta_dot_last, [self.batch_size, 1])
        upsilon_dot_last = tf.reshape(upsilon_dot_last, [self.batch_size, 3])

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

        upsilon_dot = tf.reshape(upsilon_dot, [self.batch_size, 3])
        upsilon = tf.reshape(upsilon, [self.batch_size, 3])

    
        upsilon = (self.train_dt) * (upsilon_dot + upsilon_dot_last)/2 + upsilon  # integral

        '''J = tf.stack([[tf.cos(eta[:, 2]), -tf.sin(eta[:, 2]), zeros],
                      [tf.sin(eta[:, 2]), tf.cos(eta[:, 2]), zeros],
                      [zeros, zeros, ones]])
        J = tf.transpose(J, perm=[2, 0, 1])'''

        eta_dot = upsilon[:, 2]  # transformation into local reference frame
        eta_dot = tf.reshape(eta_dot, [self.batch_size, 1])
        eta = tf.reshape(eta, [self.batch_size, 1])

        eta = (self.train_dt) * (eta_dot + eta_dot_last)/2 + eta  # integral

        eta = tf.where(tf.greater(tf.abs(eta), np.pi), (tf.math.sign(eta))*(tf.math.abs(eta)-2*np.pi), eta)

        eu = self.train_target[1] - upsilon[:, 0]

        port_vector = port[:, 1:20]
        port_vector = tf.reshape(port_vector, [self.batch_size, 19])

        stbd_vector = stbd[:, 1:20]
        stbd_vector = tf.reshape(stbd_vector, [self.batch_size, 19])

        #past_port = tf.reshape(past_port, [self.batch_size, 1])
        Tport = tf.reshape(Tport, [self.batch_size, 1])

        #past_stbd = tf.reshape(past_stbd, [self.batch_size, 1])
        Tstbd = tf.reshape(Tstbd, [self.batch_size, 1])

        std_port = tf.concat([port_vector, Tport],1)
        std_port = tf.reshape(std_port, [self.batch_size, 20])

        sigma_port = tf.math.reduce_std(std_port,1)
        sigma_port = tf.reshape(sigma_port, [self.batch_size, 1])

        std_stbd = tf.concat([stbd_vector, Tstbd],1)
        std_stbd = tf.reshape(std_stbd, [self.batch_size, 20])

        sigma_stbd = tf.math.reduce_std(std_stbd,1)
        sigma_stbd = tf.reshape(sigma_stbd, [self.batch_size, 1])

        eta = tf.reshape(eta, [self.batch_size, 1])
        upsilon = tf.reshape(upsilon, [self.batch_size, 3])
        eu = tf.reshape(eu, [self.batch_size, 1])
        next_state = tf.concat([eta, upsilon, eu, Tport, Tstbd], axis=1)
        eta_dot = tf.reshape(eta_dot, [self.batch_size, 1])
        upsilon_dot = tf.reshape(upsilon_dot, [self.batch_size, 3])
        next_last = tf.concat([eta_dot, upsilon_dot], 1)
        reward, e_psi = self.get_reward(next_state, sigma_port, sigma_stbd)
        return next_state, reward, e_psi, std_port, std_stbd, sigma_port, sigma_stbd, next_last

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

        self.ph_initial_thrusts = tf.placeholder(tf.float32, shape=[self.batch_size, 20])
        port = self.ph_initial_thrusts
        stbd = self.ph_initial_thrusts

        self.ph_initial_lasts = tf.placeholder(tf.float32, shape=[self.batch_size, 4])
        last = self.ph_initial_lasts

        # Unrolled trajectory graph
        actions = []
        rewards = []
        trajectory = [state]
        heading_errors = []
        ports = [port]
        stbds = [stbd]
        sigma_ports = []
        sigma_stbds = []
        lasts = [last]
        for t in range(self.graph_timesteps):
            action = self.run_action_network(state)
            state, reward, e_psi, port, stbd, sigma_p, sigma_s, last = self.next_timestep(state, action, port, stbd, last)
            total_reward += (self.discount_factor ** t) * reward
            actions.append(action)
            rewards.append(reward)
            trajectory.append(state)
            heading_errors.append(e_psi)
            ports.append(port)
            stbds.append(stbd)
            sigma_ports.append(sigma_p)
            sigma_stbds.append(sigma_s)
            lasts.append(last)

        self.actions = tf.stack(actions)
        self.rewards = tf.stack(rewards)
        self.trajectory = tf.stack(trajectory)
        self.e_psi = tf.stack(heading_errors)
        self.ports = tf.stack(ports)
        self.stbds = tf.stack(stbds)
        self.sigma_ports = tf.stack(sigma_ports)
        self.sigma_stbds = tf.stack(sigma_stbds)
        self.lasts = tf.stack(lasts)

        # Initialize optimizer
        self.average_total_reward = tf.reduce_mean(total_reward)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00005)
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
            ylabels = ['psi', 'u', 'v', 'r', 'e_u', 'e_psi', 'sigma_port', 'sigma_stbd',
                       'Tport', 'Tstbd', 'avg. total reward']
            axes = []
            for i in range(11):
                ax = fig.add_subplot(3, 4, i + 1)
                ax.xaxis.set_label_coords(1.05, -0.025)
                plt.xlabel('time')
                plt.ylabel(ylabels[i])
                for j in range(self.batch_size):
                    ax.plot([])
                axes.append(ax)
            plt.xlabel('iterations')

        initial_thrusts = np.zeros([self.batch_size, 20])
        initial_lasts = np.zeros([self.batch_size, 4])

        # Train
        iterations = []
        average_total_rewards = []
        for i in range(train_iterations + 1):
            
            # Get results
            actions, rewards, trajectory, heading_error, sigma_port, sigma_stbd, average_total_reward = self.sess.run(
                [self.actions, self.rewards, self.trajectory, self.e_psi, self.sigma_ports, self.sigma_stbds, self.average_total_reward],
                feed_dict={self.ph_initial_state: initial_state, self.ph_initial_thrusts: initial_thrusts, self.ph_initial_lasts: initial_lasts})

            # Update weight variables
            if i != train_iterations:
                self.sess.run(self.update, feed_dict={self.ph_initial_state: initial_state, self.ph_initial_thrusts: initial_thrusts, self.ph_initial_lasts: initial_lasts})

            # Plot results
            iterations.append(i)
            average_total_rewards.append(average_total_reward)
            print("iteration: ", i, 'Average total reward:', average_total_reward)
            if i % 50 == 0:
                print('iteration:', i, 'Average total reward:', average_total_reward)
                print("train target: ", self.train_target[1])
                if self.graphical:
                    for traj in range(self.batch_size):
                        time = np.arange(self.graph_timesteps + 1) * self.train_dt
                        for j in range(5):
                            del axes[j].lines[0]
                            axes[j].plot(time, trajectory[:, traj, j], 'b-')
                            axes[j].relim()
                        del axes[5].lines[0]
                        axes[5].plot(time[:-1], heading_error[:, traj], 'b-')
                        axes[5].relim()
                        del axes[6].lines[0]
                        axes[6].plot(time[:-1], sigma_port[:, traj], 'b-')
                        axes[6].relim()
                        del axes[7].lines[0]
                        axes[7].plot(time[:-1], sigma_stbd[:, traj], 'b-')
                        axes[7].relim()
                        for j in range(2):
                            del axes[j+8].lines[0]
                            axes[j+8].plot(time[:-1], actions[:, traj, j], 'b-')
                            axes[j+8].relim()
                    del axes[10].lines[0]
                    axes[10].plot(iterations, average_total_rewards, 'b-')
                    plt.pause(1e-10)
                

            '''if (i > 1 and i % 500 == 0):
                self.train_target = self.get_train_target()
                self.random_state = self.get_random_state()
                self.save_model('example'+ str(i)) #find a way to  '''


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
xlim = [-0, 0]
ylim = [-0, 0]
yawlim = [-np.pi, np.pi]
# Random upsilon limits
xvel_lim = [0.0, 1.5]
yvel_lim = [-0.0, 0.0]
yaw_ang_vel_lim = [-0.0, 0.0]
# Lists of parameters
eta_limits = [xlim, ylim, yawlim]
upsilon_limits = [xvel_lim, yvel_lim, yaw_ang_vel_lim]
# Create objects
total_iterations=50000
train_iterations= 500
cycles = total_iterations/train_iterations
for k in range(int(cycles)):
    n=(k)*train_iterations
    if k==0:
        model_name= None#'example'+ str(2500)
    else: 
        model_name='example'+ str(n)
    # Create objects
    boat = Boat(random_eta_limits=eta_limits, random_upsilon_limits=upsilon_limits)
    ctrl = BPTT_Controller(boat, train=True, num_hidden_units=[64, 64], graph_timesteps=250, train_dt=0.02, train_iterations=train_iterations, model_name=model_name)
    ctrl.save_model('example'+ str(n+train_iterations))
    del ctrl

print(total_iterations)
