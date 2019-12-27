import numpy as np
from physics import *

#TODO: Implement roll and pitch controllers w.r.t. quaternions instead of euler angles using angle-axis representation
#TODO: Tune the PIDs with Matlab auto-tunning toolkit

def to_rotor_speeds(hover, climb, roll, pitch, yaw):
    '''Map control commands into rotor speeds
    Params
    ======
        hover: float, hover rotor speed
        climb: float, climb command
        roll: float, roll command
        pitch: float, pitch command
        yaw: float, yaw command
    Returns
    ======
        float list, speed of each rotor in rad/s
    '''
    rotor_speeds = [None] * 4
    rotor_speeds[0] = hover + climb - pitch - yaw
    rotor_speeds[1] = hover + climb + roll  + yaw
    rotor_speeds[2] = hover + climb + pitch - yaw
    rotor_speeds[3] = hover + climb - roll  + yaw
    return rotor_speeds
    
class PID():
    '''Basic PID controller'''
    def __init__(self, kp, ki, kd, dt):
        '''Initialize PID gains and variables
        Params
        ======
            kp: float, proportional gain
            ki: float, integral gain
            kd: float, derivative gain
            dt: float, time between commands in seconds
        '''
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.reset()

    def reset(self):
        '''Reset PID variables to initial state'''
        self.integral = 0.
        self.error_prior = 0.
        self.first_iteration = True

    def output(self, error):
        '''Calculate the output PID command
        Params
        ======
            error: float, input error between desired and actual value
        Returns
        ======
            float, output PID command
        '''
        self.integral += error * self.dt
        if self.first_iteration: # Avoid big derivatives at first iteration
            derivative = 0.
            self.first_iteration = False
        else:
            derivative = (error - self.error_prior) / self.dt
        self.error_prior = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output
        
class Cascade_PID_Controller():
    '''Implements a cascade PID controller with four inner loop PIDs for
    controlling climb, roll, pitch and yaw commands; and thwo outer loop
    PIDs for controlling roll and pitch angles'''
    def __init__(self, quad):
        '''Initialize all six PIDs
        Params
        ======
            quad: Quadcopter object
        '''
        self.quad = quad

        # Inner loop PIDs
        climb_kp = 12.5
        climb_ki = 0.
        climb_kd = 30.
        self.climb_pid = PID(climb_kp, climb_ki, climb_kd, self.quad.physics.dt)
        roll_kp = 7.
        roll_ki = 0.
        roll_kd = 6.6
        self.roll_pid = PID(roll_kp, roll_ki, roll_kd, self.quad.physics.dt)
        pitch_kp = 7.
        pitch_ki = 0.
        pitch_kd = 6.6
        self.pitch_pid = PID(pitch_kp, pitch_ki, pitch_kd, self.quad.physics.dt)
        yaw_kp = 6.
        yaw_ki = 0.
        yaw_kd = 14.3
        self.yaw_pid = PID(yaw_kp, yaw_ki, yaw_kd, self.quad.physics.dt)

        # Outer loop PIDs
        target_roll_kp = -0.059
        target_roll_ki = -0.0
        target_roll_kd = -0.1
        self.target_roll_pid = PID(target_roll_kp, target_roll_ki, target_roll_kd, self.quad.physics.dt)
        target_pitch_kp = 0.059
        target_pitch_ki = 0.000
        target_pitch_kd = 0.1
        self.target_pitch_pid = PID(target_pitch_kp, target_pitch_ki, target_pitch_kd, self.quad.physics.dt)

        self.reset()

    def reset(self):
        '''Reset PIDs to initial state'''
        self.climb_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
        self.target_roll_pid.reset()
        self.target_pitch_pid.reset()

    def position_control(self, target=None):
        '''Calculate rotor speeds to stabilize
        or to reach a target if preset
        Params
        ======
            target: 1D numpy array, target position in x-y-z coordinates
        Returns
        ======
            float list, speed for each rotor in rad/s
        '''
        # Euler angles
        eul_ang = quaternion_to_euler_angles(*self.quad.physics.orientation)
        sph = np.sin(eul_ang[0])
        sth = np.sin(eul_ang[1])
        cph = np.cos(eul_ang[0])
        cth = np.cos(eul_ang[1])

        # Rotation matrix
        R = inertial_to_body_frame(*self.quad.physics.orientation)

        # Outer loop controll
        if target is not None:
            target_error = target - self.quad.physics.position
            target_error = np.matmul(R, target_error)
            target_roll_error = target_error[1] * cph - target_error[2] * sph
            target_pitch_error = target_error[0] * cth + target_error[2] * sth
        else:
            target_roll_error = -self.quad.physics.vel[1]
            target_pitch_error = -self.quad.physics.vel[0]
            
        target_roll = self.target_roll_pid.output(target_roll_error)
        target_pitch = self.target_pitch_pid.output(target_pitch_error)
        target_roll = min(0.25 * np.pi, target_roll)
        target_roll = max(-0.25 * np.pi, target_roll)
        target_pitch = min(0.25 * np.pi, target_pitch)
        target_pitch = max(-0.25 * np.pi, target_pitch)

        # Inner loop control
        if target is not None:
            climb_error = target[2] - self.quad.physics.position[2]
        else:
            climb_error = -self.quad.physics.vel[2]
        roll_error = target_roll - eul_ang[0]
        pitch_error = target_pitch - eul_ang[1]
        yaw_error = 0. - eul_ang[2]
        
        climb = self.climb_pid.output(climb_error)
        roll = self.roll_pid.output(roll_error)
        pitch = self.pitch_pid.output(pitch_error)
        yaw = self.yaw_pid.output(yaw_error)

        # Rotor speeds
        rotor_speeds = to_rotor_speeds(self.quad.physics.hover_rotor_speed,
                                       climb, roll, pitch, yaw)
        rotor_speeds = [np.minimum(1000, rs) for rs in rotor_speeds]
        rotor_speeds = [np.maximum(0, rs) for rs in rotor_speeds]
        return rotor_speeds
