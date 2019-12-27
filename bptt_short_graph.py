import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from controller import to_rotor_speeds

def quaternion_to_euler_angles(orientation_timeseries):
    '''Convert orientation quaternion into euler angles
    Size of numpy arrays' second dimension is the batch size for parallel computation
    Params
    ======
        orientation_timeseries: 3D numpy array, orientation quaternion timeseries
    Returns
    ======
        3D numpy array, euler angles timeseires
    '''
    w = orientation_timeseries[:,:,0]
    x = orientation_timeseries[:,:,1]
    y = orientation_timeseries[:,:,2]
    z = orientation_timeseries[:,:,3]
    phi = np.arctan2(2 * (y * z + w * x), w * w - x * x - y * y + z * z)
    sinp = 2 * (x * z - w * y)
    theta = np.where(np.absolute(sinp) >= 1.0, np.copysign(0.5 * np.pi, -sinp), -np.arcsin(sinp))
    psi = np.arctan2(2 * (x * y + w * z), w * w + x * x - y * y - z * z)
    return np.stack([phi, theta, psi], axis=2)

def quaternion_vector_mult(q, v):
    '''Calculate the quaternion-vector product q * v
    v is a "pure" quaternion, with real part equal to zero
    Params
    ======
        q: rank-2 tensor, quaternion
        v: rank-2 tensor, vector
    Returns
    ======
        rank-2 tensor, quaternion-vector product q * v
    '''
    qw = q[:,0]
    qx = q[:,1]
    qy = q[:,2]
    qz = q[:,3]
    vx = v[:,0]
    vy = v[:,1]
    vz = v[:,2]
    w = - qx * vx - qy * vy - qz * vz
    x = qw * vx + qy * vz - qz * vy
    y = qw * vy - qx * vz + qz * vx
    z = qw * vz + qx * vy - qy * vx
    return tf.stack([w, x, y, z], axis=1)

def body_to_inertial_frame(orientation):
    '''Rotation matrix from body frame to inertial frame using orientation quaternion
    Size of tensors' first dimension is the batch size for parallel computation
    Params
    ======
        orientation: rank-2 tensor, orientation quaternion
    Returns
    ======
        rank-3 tensor, rotation matrix from body frame to inertial frame
    '''
    w = orientation[:,0]
    x = orientation[:,1]
    y = orientation[:,2]
    z = orientation[:,3]
    R = tf.stack(
        [[1-2*(y*y+z*z),  2*(x*y-w*z),   2*(x*z+w*y)],
         [  2*(x*y+w*z),1-2*(x*x+z*z),   2*(y*z-w*x)],
         [  2*(x*z-w*y),  2*(y*z+w*x),1-2*(x*x+y*y)]])
    return tf.transpose(R, perm=[2,1,0])

def inertial_to_body_frame(orientation):
    '''Rotation matrix from inertial frame to body frame using orientation quaternion
    Size of tensors' first dimension is the batch size for parallel computation
    Params
    ======
        orientation: rank-2 tensor, orientation quaternion
    Returns
    ======
        rank-3 tensor, rotation matrix from inertial frame to body frame
    '''
    w = orientation[:,0]
    x = orientation[:,1]
    y = orientation[:,2]
    z = orientation[:,3]
    R = tf.stack(
        [[1-2*(y*y+z*z),  2*(x*y+w*z),  2*(x*z-w*y)],
         [  2*(x*y-w*z),1-2*(x*x+z*z),  2*(y*z+w*x)],
         [  2*(x*z+w*y),  2*(y*z-w*x),1-2*(x*x+y*y)]])
    return tf.transpose(R, perm=[2,1,0])

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
    Backpropagation Through Time using a short graph in series
    instead of using a single long graph'''
    def __init__(self, quad, train=False, num_hidden_units=[8,8], batch_size=10,
                 graph_timesteps=20, discount_factor=1.0, train_dt=0.025, train_runtime=5.0,
                 train_iterations=1000, model_name=None, graphical=True):
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
            train_runtime: float, simulation runtime for training in seconds
            train_iterations: int, train iterations
            model_name: string, name of the model to restore
            graphical: bool, display graph during training
        '''
        self.quad = quad
        self.train_dt = train_dt
        self.train_runtime = train_runtime
        self.graphical = graphical
        
        # Neural network parameters
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.action_network_num_inputs = self.quad.state.size
        self.action_network_num_outputs = 4
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
        train_target = self.quad.random_position_limits.copy()
        for i in range(3):
            train_target[i] = train_target[i][0] + train_target[i][1]
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
            self.quad.reset_random()
            state = self.quad.state.copy()
            for j in range(3):
                state[j] *= 2.
            random_state.append(state)
        return np.array(random_state)
    
    def reset_random(self):
        '''Reset parameters to random values and
        start a new session'''
        tf.reset_default_graph()
        
        self.W1 = weight_variable([self.action_network_num_inputs, self.num_hidden_units[0]])
        self.b1 = weight_variable([1, self.num_hidden_units[0]])
        self.W2 = weight_variable([self.action_network_num_inputs + self.num_hidden_units[0], self.num_hidden_units[1]])
        self.b2 = weight_variable([1, self.num_hidden_units[1]])
        self.W3 = weight_variable([self.action_network_num_inputs + self.num_hidden_units[0] + self.num_hidden_units[1], self.action_network_num_outputs])
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
        return action

    def get_reward(self, state):
        '''Calculate the reward given a state
        Size of tensors' first dimension is the batch size for parallel computation
        Params
        ======
            state: rank-2 tensor, state in the form [position,orientation,vel,ang_vel]
        Returns
        ======
            rank-1 tensor, reward
        '''
        position = state[:,0:3]
        orientation_vector = state[:,4:7]
        # Distance between train target and current position
        reward = -tf.sqrt(tf.reduce_sum(tf.square(self.train_target - position), axis=1))
        reward -= tf.norm(orientation_vector, axis=1)
        return reward

    def next_timestep(self, state, action):
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
        position = state[:,0:3]
        orientation = state[:,3:7]
        vel = state[:,7:10]
        ang_vel = state[:,10:13]
        action = [action[:,0], action[:,1], action[:,2], action[:,3]]
        
        # Body frame
        rotor_speeds = to_rotor_speeds(self.quad.physics.hover_rotor_speed, *action)
        rotor_speeds = tf.stack(rotor_speeds, axis=1)
        thrust_and_torque = tf.matmul(tf.square(rotor_speeds), tf.cast(tf.transpose(self.quad.physics.thrust_and_torque_constants), tf.float32))
            
        zeros = tf.zeros([self.batch_size], dtype=tf.float32)
        thrust = thrust_and_torque[:,0]
        f_prop = tf.stack([zeros, zeros, thrust], axis=1)
        grav = tf.constant(self.quad.physics.gravity, dtype=tf.float32, shape=[self.batch_size])
        grav = tf.stack([zeros, zeros, grav], axis=1)
        grav = tf.reshape(grav, [self.batch_size, 1, 3])
        grav = tf.matmul(grav, inertial_to_body_frame(orientation))
        grav = tf.reshape(grav, [self.batch_size, 3])
        acc = 1 / self.quad.physics.mass * f_prop + grav - tf.cross(ang_vel, vel) # Adding coriolis acceleration
        vel += acc * self.train_dt

        torque = thrust_and_torque[:,1:4]
        ang_acc = torque - tf.cross(ang_vel, tf.cast(self.quad.physics.inertia_moments, tf.float32) * ang_vel) # Adding coriolis angular acceleration
        ang_acc = tf.cast(1 / self.quad.physics.inertia_moments, tf.float32) * ang_acc
        ang_vel += ang_acc * self.train_dt
            
        # Inertial frame
        vel_inertial_frame = tf.reshape(vel, [self.batch_size, 1, 3])
        vel_inertial_frame = tf.matmul(vel_inertial_frame, body_to_inertial_frame(orientation))
        vel_inertial_frame = tf.reshape(vel_inertial_frame, [self.batch_size, 3])
        position += vel_inertial_frame * self.train_dt
        orientation += 0.5 * quaternion_vector_mult(orientation, ang_vel) * self.train_dt
        orientation = tf.transpose(tf.transpose(orientation) / tf.norm(orientation, axis=1))
                
        next_state = tf.concat([position, orientation, vel, ang_vel], axis=1)
        reward = self.get_reward(next_state)
        return next_state, reward

    def build_graph(self, graph_timesteps):
        '''Build a tensorflow graph to train using an optimizer
        Params
        ======
            graph_timesteps: int, timesteps for the tensorflow graph
        '''
        self.reset_random()
        self.graph_timesteps = graph_timesteps
        total_reward = tf.zeros([self.batch_size], dtype=tf.float32)

        # Placeholders
        self.ph_initial_state = tf.placeholder(tf.float32, shape=[self.batch_size, self.action_network_num_inputs])
        self.ph_grad_state = tf.placeholder(tf.float32, shape=self.ph_initial_state.shape)
        weight_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.ph_total_grad_weight_list = [tf.placeholder(tf.float32, shape=w.shape) for w in weight_list]
        state = self.ph_initial_state
        
        # Unrolled trajectory graph
        actions = []
        rewards = []
        trajectory = [state]
        for t in range(self.graph_timesteps):
            action = self.run_action_network(state)
            state, reward = self.next_timestep(state, action)
            total_reward += (self.discount_factor ** t) * reward
            actions.append(action)
            rewards.append(reward)
            trajectory.append(state)
        
        self.actions = tf.stack(actions)
        self.rewards = tf.stack(rewards)
        self.trajectory = tf.stack(trajectory)

        # Initialize optimizer
        optimizer = tf.train.AdamOptimizer()
        
        # Compute gradients and new weights
        self.average_total_reward = tf.reduce_mean(total_reward)
        grad_ys = [tf.ones_like(total_reward), self.ph_grad_state]
        gradients = tf.gradients([total_reward, (self.discount_factor ** self.graph_timesteps) * state],
                                 [self.ph_initial_state] + weight_list, grad_ys=grad_ys)
        self.grad_state = gradients[0]
        self.grad_weight_list = gradients[1:]
        self.update = optimizer.apply_gradients(zip(self.ph_total_grad_weight_list, weight_list))
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
            ylabels = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'climb',
                       'roll', 'pitch', 'yaw', 'reward', 'avg. total reward']
            axes = []
            for i in range(12):
                ax = fig.add_subplot(4, 3, i + 1)
                ax.xaxis.set_label_coords(1.05, -0.025)
                plt.xlabel('time')
                plt.ylabel(ylabels[i])
                for j in range(self.batch_size):
                    ax.plot([])
                axes.append(ax)
            plt.xlabel('iterations')

        # Train
        iterations = []
        average_total_rewards = []
        for i in range(train_iterations + 1):
            # Get results
            actions = []
            rewards = []
            trajectory = []
            state = initial_state
            average_total_reward = 0.0
            
            # Run the graph num_substeps times
            num_substeps = round(self.train_runtime / (self.graph_timesteps * self.train_dt))
            for substep in range(num_substeps):
                actions_substep, rewards_substep, trajectory_substep, average_total_reward_substep = self.sess.run(
                    [self.actions, self.rewards, self.trajectory, self.average_total_reward],
                    feed_dict={self.ph_initial_state: state})
                actions.append(actions_substep)
                rewards.append(rewards_substep)
                trajectory.append(trajectory_substep[:-1])
                state = trajectory_substep[-1]
                average_total_reward += (self.discount_factor ** (substep * self.graph_timesteps)) * average_total_reward_substep

            actions = np.concatenate(actions)
            rewards = np.concatenate(rewards)
            trajectory[-1] = np.concatenate((trajectory[-1], np.array([state])))
            trajectory = np.concatenate(trajectory)
            
            # Back propagation through time
            grad_state = np.zeros_like(state)
            total_grad_weight_list = [np.zeros_like(grad_w) for grad_w in self.grad_weight_list]
            time_step = len(trajectory) - self.graph_timesteps - 1
            while time_step >= 0:
                state = trajectory[time_step]
                grad_state, grad_weight_list = self.sess.run([self.grad_state, self.grad_weight_list],
                                                 feed_dict={self.ph_initial_state: state,
                                                            self.ph_grad_state: grad_state})
                total_grad_weight_list = [(self.discount_factor ** self.graph_timesteps) * total_grad_w + grad_w
                                          for total_grad_w, grad_w in zip(total_grad_weight_list, grad_weight_list)]
                time_step -= self.graph_timesteps
            total_grad_weight_list = [-total_grad_w / self.batch_size for total_grad_w in total_grad_weight_list]
            
            # Update weight variables
            if i != train_iterations:
                self.sess.run(self.update, feed_dict=dict(zip(self.ph_total_grad_weight_list, total_grad_weight_list)))

            # Plot results
            iterations.append(i)
            average_total_rewards.append(average_total_reward)
            if i % 100 == 0:
                print('iteration:', i, 'Average total reward:', average_total_reward)
                if self.graphical:
                    euler_angles = quaternion_to_euler_angles(trajectory[:,:,3:7])
                    for traj in range(self.batch_size):
                        time = np.arange(self.graph_timesteps * num_substeps + 1) * self.train_dt
                        for j in range(3):
                            del axes[j].lines[0]
                            axes[j].plot(time, trajectory[:,traj,j], 'b-')
                            axes[j].relim()
                        for j in range(3):
                            del axes[j+3].lines[0]
                            axes[j+3].plot(time, euler_angles[:,traj,j], 'b-')
                            axes[j+3].relim()
                        for j in range(4):
                            del axes[j+6].lines[0]
                            axes[j+6].plot(time[:-1], actions[:,traj,j], 'b-')
                            axes[j+6].relim()
                        del axes[10].lines[0]
                        axes[10].plot(time[:-1], rewards[:,traj], 'b-')
                        axes[10].relim()
                    del axes[11].lines[0]
                    axes[11].plot(iterations, average_total_rewards, 'b-')
                    plt.pause(1e-10)

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
        state = self.quad.state.copy()
        state[0:3] -= (target - self.train_target)
        action = self.run_numpy_action_network(state)
        rotor_speeds = to_rotor_speeds(self.quad.physics.hover_rotor_speed, *action)
        return rotor_speeds

    def reset(self):
        '''Used in simulation.py to reset the parameters of the contrroler
        between simulations (e.g. initial error and integral in a PID controller)
        It is not necessary to implement for a trained neural network'''
        pass

'''To train'''
'''
from quadcopter import Quadcopter
# Graph3D limits
xlim = [-2.5,2.5]
ylim = [-2.5,2.5]
zlim = [0.0,5.0]
# Random state generator limits
tilt_ang_lim = 0. * np.pi # rad
yaw_ang_lim = 0. * np.pi # rad
vel_lim = 0. # m/s
ang_vel_lim = 0. * np.pi # rad/s
# Lists of parameters
graph_limits = [xlim, ylim, zlim]
random_state_limits = [tilt_ang_lim, yaw_ang_lim, vel_lim, ang_vel_lim]
# Create objects
quad = Quadcopter(random_position_limits=graph_limits,
                  random_state_limits=random_state_limits)
ctrl = BPTT_Controller(quad, train=True, num_hidden_units=[64,64],
                       graph_timesteps=20, train_dt=0.025, train_runtime=5.0,
                       train_iterations=1000)
ctrl.save_model('example')
'''
