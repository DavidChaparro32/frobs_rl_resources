#!/bin/python3

import gym
from gym import utils
from gym import spaces
from gym.envs.registration import register
from frobs_rl.common import ros_gazebo
from frobs_rl.common import ros_controllers
from frobs_rl.common import ros_node
from frobs_rl.common import ros_launch
from frobs_rl.common import ros_params
from frobs_rl.common import ros_urdf
from frobs_rl.common import ros_spawn
from abb_irb120_reacher.robot_env import abb_irb120_moveit
import rospy
import rostopic
import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_matrix, quaternion_from_matrix, rotation_from_matrix
from angles import normalize_angle

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from gazebo_msgs.srv import SetLinkState, SetLinkStateRequest, SetLinkStateResponse

import numpy as np
import scipy.spatial

register(
        id='ABBIRB120ReacherEnv-v0',
        entry_point='abb_irb120_reacher.task_env.irb120_reacher:ABBIRB120ReacherEnv',
        max_episode_steps=10000
    )

class ABBIRB120ReacherEnv(abb_irb120_moveit.ABBIRB120MoveItEnv):
    """
    Custom Task Env, use this env to implement a task using the robot defined in the CustomRobotEnv
    """

    def __init__(self):
        """
        Describe the task.
        """
        rospy.logwarn("Starting ABBIRB120ReacherEnv Task Env")

        """
        Load YAML param file
        """
        ros_params.ros_load_yaml_from_pkg("abb_irb120_reacher", "reacher_task.yaml", ns="/")
        self.get_params()

        """
        Define the action and observation space.
        """
        #--- Define the ACTION SPACE
        # Define a continuous space using BOX and defining its limits
        self.action_space = spaces.Box(low=np.array(self.min_joint_values), high=np.array(self.max_joint_values), dtype=np.float32)

        #--- Define the OBSERVATION SPACE

        #- Define the maximum and minimum pose allowed for the EE
        obsrv_high_ee_pos_range = np.array(np.array([self.position_ee_max["x"], self.position_ee_max["y"], self.position_ee_max["z"]]))
        obsrv_low_ee_pos_range  = np.array(np.array([self.position_ee_min["x"], self.position_ee_min["y"], self.position_ee_min["z"]]))
        obsrv_high_ee_ori_range = np.array([ 1.0,  1.0,  1.0,  1.0])
        obsrv_low_ee_ori_range  = np.array([-1.0, -1.0, -1.0, -1.0])

        obsrv_high_ee = np.concatenate((obsrv_high_ee_pos_range, obsrv_high_ee_ori_range))
        obsrv_low_ee  = np.concatenate((obsrv_low_ee_pos_range, obsrv_low_ee_ori_range))

        #- Define the maximum and minimum pose allowed for the goal
        obsrv_high_goal_pos_range = np.array(np.array([self.position_goal_max["x"], self.position_goal_max["y"], self.position_goal_max["z"]]))
        obsrv_low_goal_pos_range  = np.array(np.array([self.position_goal_min["x"], self.position_goal_min["y"], self.position_goal_min["z"]]))
        obsrv_high_goal_ori_range = np.array([ 1.0,  1.0,  1.0,  1.0])
        obsrv_low_goal_ori_range  = np.array([-1.0, -1.0, -1.0, -1.0])

        obsrv_high_goal = np.concatenate((obsrv_high_goal_pos_range, obsrv_high_goal_ori_range))
        obsrv_low_goal  = np.concatenate((obsrv_low_goal_pos_range, obsrv_low_goal_ori_range))

        #- Define the range for the unit vector from the EE to the goal
        obsrv_high_vec_EE_GOAL = np.array([1.0, 1.0, 1.0])
        obsrv_low_vec_EE_GOAL  = np.array([-1.0, -1.0, -1.0])

        #--- Concatenate the observation space limits for positions and distance to goal
        high = np.concatenate([self.max_joint_values, obsrv_high_ee, obsrv_high_goal, obsrv_high_vec_EE_GOAL, obsrv_high_vec_EE_GOAL])
        low  = np.concatenate([self.min_joint_values, obsrv_low_ee , obsrv_low_goal,  obsrv_low_vec_EE_GOAL,  obsrv_low_vec_EE_GOAL])

        #--- Observation space
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32) 

        """
        Define subscribers or publishers as needed.
        """

        #--- Make Marker msg for publishing
        self.goal_marker = Marker()
        self.goal_marker.header.frame_id="world"
        self.goal_marker.header.stamp = rospy.Time.now()
        self.goal_marker.ns = "goal_shapes"
        self.goal_marker.id = 0
        self.goal_marker.type = Marker.SPHERE
        self.goal_marker.action = Marker.ADD

        self.goal_marker.pose.position.x = 0.0
        self.goal_marker.pose.position.y = 0.0
        self.goal_marker.pose.position.z = 0.0
        self.goal_marker.pose.orientation.x = 0.0
        self.goal_marker.pose.orientation.y = 0.0
        self.goal_marker.pose.orientation.z = 0.0
        self.goal_marker.pose.orientation.w = 1.0

        self.goal_marker.scale.x = 0.1
        self.goal_marker.scale.y = 0.1
        self.goal_marker.scale.z = 0.1

        self.goal_marker.color.r = 1.0
        self.goal_marker.color.g = 0.0
        self.goal_marker.color.b = 0.0
        self.goal_marker.color.a = 1.0

        self.pub_marker = rospy.Publisher("goal_point",Marker,queue_size=10)

        #--- Publish transform
        self.goal_pos = np.array([0.0, 0.0, 0.0])
        self.goal_ori = np.array([0.0, 0.0, 0.0, 1.0])
        self.tf_br = tf.TransformBroadcaster()

        self.tf_br.sendTransform(self.goal_pos, self.goal_ori,
                        rospy.Time.now()+rospy.Duration(3.0),
                        "goal_frame",
                        "world")


        self.goal_subs  = rospy.Subscriber("goal_pos", Point, self.goal_callback)
        if self.training:
            ros_node.ros_node_from_pkg("abb_irb120_reacher", "pos_publisher.py", name="pos_publisher", ns="/")
            rospy.wait_for_service("set_init_point")
            self.set_init_goal_client = rospy.ServiceProxy("set_init_point", SetLinkState)

        """
        Init super class.
        """
        super(ABBIRB120ReacherEnv, self).__init__()

        """
        Finished __init__ method
        """
        rospy.logwarn("Finished Init of ABBIRB120ReacherEnv Task Env")

    #-------------------------------------------------------#
    #   Custom available methods for the CustomTaskEnv      #

    def _set_episode_init_params(self):
        """
        Sets the Robot in its init pose
        The Simulation will be unpaused for this purpose.
        """

        self.init_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = self.set_trajectory_joints(self.init_pos)
        if not result:
            rospy.logwarn("Homing has failed....")

        #--- If training set random goal
        if self.training:
            self.init_pos = self.get_randomJointVals()
            self.goal_pos, self.goal_ori = self.get_randomValidGoal()

            init_goal_msg = SetLinkStateRequest()
            init_goal_msg.link_state.pose.position.x = self.goal_pos[0]
            init_goal_msg.link_state.pose.position.y = self.goal_pos[1]
            init_goal_msg.link_state.pose.position.z = self.goal_pos[2]

            self.set_init_goal_client.call(init_goal_msg)
            rospy.logwarn("Desired goal--->" + str(self.goal_pos) + " " + str(self.goal_ori))

        #--- Make Marker msg for publishing
        self.goal_marker.pose.position.x = self.goal_pos[0]
        self.goal_marker.pose.position.y = self.goal_pos[1]
        self.goal_marker.pose.position.z = self.goal_pos[2]

        self.goal_marker.color.r = 1.0
        self.goal_marker.color.g = 0.0
        self.goal_marker.color.b = 0.0
        self.goal_marker.color.a = 1.0

        self.goal_marker.lifetime = rospy.Duration(secs=30)
        
        self.pub_marker.publish(self.goal_marker)

        self.tf_br.sendTransform(self.goal_pos, self.goal_ori,
                        rospy.Time.now()+rospy.Duration(3.0),
                        "goal_frame",
                        "world")

        #--- Set the initial joint values
        rospy.logwarn("Initializing with values" + str(self.init_pos))
        result = self.set_trajectory_joints(self.init_pos)
        self.joint_angles = self.init_pos
        if not result:
            rospy.logwarn("Initialisation is failed....")

    def _send_action(self, action):
        """
        The action are the joint positions
        """
        rospy.logwarn("=== Action: {}".format(action))

        #--- Make actions as deltas
        action = self.joint_values + action
        action = np.clip(action, self.min_joint_values, self.max_joint_values)

        self.movement_result = self.set_trajectory_joints(action)
        if not self.movement_result:
            rospy.logwarn("Movement_result failed with the action of : " + str(action))
        

    def _get_observation(self):
        """
        It returns the position of the EndEffector as observation.
        And the distance from the desired point
        Orientation for the moment is not considered
        TODO Check if observations are enough
        """

        #--- Get Current Joint values
        self.joint_values = self.get_joint_angles()

        #--- Get EE pose
        ee_pose = self.get_ee_pose() # Get a geometry_msgs/PoseStamped msg

        self.ee_pos = np.array([ee_pose.pose.position.x, ee_pose.pose.position.y, ee_pose.pose.position.z])
        self.ee_ori = np.array([ee_pose.pose.orientation.x, ee_pose.pose.orientation.y, ee_pose.pose.orientation.z, ee_pose.pose.orientation.w])

        #--- Get current goal
        current_goal_pos = self.goal_pos
        current_goal_ori = self.goal_ori

        #--- Normalized Vector to goal
        vec_EE_GOAL_pos = current_goal_pos - self.ee_pos
        vec_EE_GOAL_pos = vec_EE_GOAL_pos / np.linalg.norm(vec_EE_GOAL_pos)

        #--- Orientation error vector
        vec_EE_GOAL_ori = self.calc_ori_error_2(current_goal_ori, self.ee_ori) # TODO Check both error function

        obs = np.concatenate((
            self.joint_values,       # Current joint angles
            self.ee_pos,             # Current position of EE
            self.ee_ori,             # Current orientation of EE
            current_goal_pos,        # Position of Goal
            current_goal_ori,        # Orientation of Goal
            vec_EE_GOAL_pos,         # Normalized Distance Vector from EE to Goal
            vec_EE_GOAL_ori,         # Orientation error vector
            ),
            axis=None
        )

        rospy.logwarn("OBSERVATIONS====>>>>>>>"+str(obs))

        return obs.copy()

    def _get_reward(self):
        """
        Given a success of the execution of the action
        Calculate the reward: binary => 1 for success, 0 for failure
        TODO give reward if current distance is lower than previous
        """

        #--- Get current EE pos 
        current_pos = self.ee_pos 
        current_ori = self.ee_ori

        #--- Get current goal
        current_goal_pos = self.goal_pos
        current_goal_ori = self.goal_ori

        #- Init reward
        reward = 0

        #- Check if the EE reached the goal
        done = False
        done = self.calculate_if_done(self.movement_result, current_goal_pos, current_pos, current_goal_ori, current_ori)
        if done:
            if self.pos_dynamic is False:
                rospy.logwarn("SUCCESS Reached a Desired Position!")
                self.info['is_success'] = 1.0

            #- Success reward
            reward += self.reached_goal_reward

            # Publish goal_marker 
            self.goal_marker.header.stamp = rospy.Time.now()
            self.goal_marker.color.r = 0.0
            self.goal_marker.color.g = 1.0
            self.goal_marker.lifetime = rospy.Duration(secs=5)
        else:
            # Publish goal_marker 
            self.goal_marker.header.stamp = rospy.Time.now()
            self.goal_marker.color.r = 1.0
            self.goal_marker.color.g = 0.0
            self.goal_marker.lifetime = rospy.Duration(secs=5)

            #- Distance from EE to Goal reward
            dist2goal = scipy.spatial.distance.euclidean(current_pos, current_goal_pos)
            rospy.loginfo("Pos error: " + str(dist2goal))
            if dist2goal<=self.tol_goal_pos:
                reward   += self.reached_goal_reward/2.0
            else:
                reward   += -self.mult_dist_reward*dist2goal 

            #- Orientation error reward
            ori_error = self.calc_ori_error_angle(current_goal_ori, current_ori)
            rospy.loginfo("Ori error: " + str(ori_error))
            # if ori_error<=self.tol_goal_ori:
            #     reward   += self.reached_goal_reward/2.0
            # else:
            reward   += -self.mult_ori_reward*ori_error 

            #- Constant reward
            reward += self.step_reward

        self.pub_marker.publish(self.goal_marker)
        self.tf_br.sendTransform(self.goal_pos, self.goal_ori,
                        rospy.Time.now()+rospy.Duration(3.0),
                        "goal_frame",
                        "world")

        #- Check if joints are in limits
        joint_angles = np.array(self.joint_values)
        min_joint_values = np.array(self.min_joint_values)
        max_joint_values = np.array(self.max_joint_values)
        in_limits = np.any(joint_angles<=(min_joint_values+0.0001)) or np.any(joint_angles>=(max_joint_values-0.0001))
        if in_limits:
            rospy.logwarn("Joints limits violated")
        reward += in_limits*self.joint_limits_reward

        rospy.logwarn(">>>REWARD>>>"+str(reward))

        return reward
    
    def _check_if_done(self):
        """
        Check if the EE is close enough to the goal
        """

        #--- Get current EE pos 
        current_pos = self.ee_pos 
        current_ori = self.ee_ori

        #--- Get current goal
        current_goal_pos = self.goal_pos
        current_goal_ori = self.goal_ori

        #--- Function used to calculate 
        done = self.calculate_if_done(self.movement_result, current_goal_pos, current_pos, current_goal_ori, current_ori)
        if done:
            rospy.logdebug("Reached a Desired Position!")

        #--- If the position is dynamic the episode is never done
        if self.pos_dynamic is True:
            done = False

        return done

    #-------------------------------------------------------#
    #  Internal methods for the ABBIRB120ReacherEnv         #

    def get_params(self):
        """
        get configuration parameters
        """
        
        self.sim_time = rospy.get_time()
        self.n_actions = rospy.get_param('/irb120/n_actions')
        self.n_observations = rospy.get_param('/irb120/n_observations')

        #--- Get parameter associated with ACTION SPACE

        self.min_joint_values = rospy.get_param('/irb120/min_joint_pos')
        self.max_joint_values = rospy.get_param('/irb120/max_joint_pos')

        assert len(self.min_joint_values) == self.n_actions , "The min joint values do not have the same size as n_actions"
        assert len(self.max_joint_values) == self.n_actions , "The max joint values do not have the same size as n_actions"

        #--- Get parameter associated with OBSERVATION SPACE

        self.position_ee_max = rospy.get_param('/irb120/position_ee_max')
        self.position_ee_min = rospy.get_param('/irb120/position_ee_min')
        self.position_goal_max = rospy.get_param('/irb120/position_goal_max')
        self.position_goal_min = rospy.get_param('/irb120/position_goal_min')
        self.max_distance = rospy.get_param('/irb120/max_distance')

        #--- Get parameter asociated to goal tolerance
        self.tol_goal_pos = rospy.get_param('/irb120/tolerance_goal_pos')
        self.tol_goal_ori = rospy.get_param('/irb120/tolerance_goal_ori')
        self.training = rospy.get_param('/irb120/training')
        self.pos_dynamic = rospy.get_param('/irb120/pos_dynamic')
        rospy.logwarn("Dynamic position:  " + str(self.pos_dynamic))

        #--- Get reward parameters
        self.reached_goal_reward = rospy.get_param('/irb120/reached_goal_reward')
        self.step_reward = rospy.get_param('/irb120/step_reward')
        self.mult_dist_reward = rospy.get_param('/irb120/multiplier_dist_reward')
        self.mult_ori_reward = rospy.get_param('/irb120/multiplier_ori_reward')
        self.joint_limits_reward = rospy.get_param('/irb120/joint_limits_reward')

        #--- Get Gazebo physics parameters
        if rospy.has_param('/irb120/time_step'):
            self.t_step = rospy.get_param('/irb120/time_step')
            ros_gazebo.gazebo_set_time_step(self.t_step)

        if rospy.has_param('/irb120/update_rate_multiplier'):
            self.max_update_rate = rospy.get_param('/irb120/update_rate_multiplier')
            ros_gazebo.gazebo_set_max_update_rate(self.max_update_rate)


    def calculate_if_done(self, movement_result, goal_pos, current_pos, goal_ori, current_ori):
        """
        It calculated whether it has finished or not
        """
        done = False

        # If the previous movement was succesful
        if movement_result:
            rospy.logdebug("Movement was succesful")
        else:
            rospy.logwarn("Movement not succesful")

        # check if the end-effector located within a threshold to the goal
        distance_2_goal   = scipy.spatial.distance.euclidean(current_pos, goal_pos)
        orientation_error = self.calc_ori_error_angle(goal_ori, current_ori)

        if distance_2_goal<=self.tol_goal_pos and orientation_error<=self.tol_goal_ori:
            done = True
        
        return done

    def goal_callback(self, data):
        """
        Callback to the topic used to send goals
        """
        self.goal_pos= np.array([data.x, data.y, data.z])

        #--- Publish goal marker
        self.goal_marker.pose.position.x = self.goal_pos[0]
        self.goal_marker.pose.position.y = self.goal_pos[1]
        self.goal_marker.pose.position.z = self.goal_pos[2]
        self.goal_marker.lifetime = rospy.Duration(secs=1)
        self.pub_marker.publish(self.goal_marker)
        self.tf_br.sendTransform(self.goal_pos, self.goal_ori,
                        rospy.Time.now()+rospy.Duration(3.0),
                        "goal_frame",
                        "world")


    def calc_ori_error(self, goal_ori, current_ori):
        """
        Calculate the orientation error Caccavale Quaternion
        """
        Rot_matrix_current = quaternion_matrix(current_ori)
        Rot_matrix_goal    = quaternion_matrix(goal_ori)

        Rot_matrix_error   = np.matmul(Rot_matrix_goal, np.transpose(Rot_matrix_current))

        quat_error = quaternion_from_matrix(Rot_matrix_error)

        ori_error = 2.0 * quat_error[3] * np.array([quat_error[0], quat_error[1], quat_error[2]])

        ori_error = ori_error / np.linalg.norm(ori_error)

        return ori_error


    def calc_ori_error_2(self, goal_ori, current_ori):
        """
        Calculate the orientation error
                error = current.w*goal[x,y,z] - goal.w*current[x,y,z] - skew(goal[x,y,z])*current[x,y,z]
        """
        skew_sym_matrix_goal = np.array([[   0,        -goal_ori[2], goal_ori[1]],
                                        [goal_ori[2],       0,      -goal_ori[0]],
                                        [-goal_ori[1],  goal_ori[0],      0]])

        ori_error = current_ori[3]*goal_ori[0:3] - goal_ori[3]*current_ori[0:3] - np.matmul(skew_sym_matrix_goal, current_ori[0:3])
        # ori_error = goal_ori[3]*current_ori[0:3] - current_ori[3]*goal_ori[0:3] - np.matmul(skew_sym_matrix_goal, current_ori[0:3])

        ori_error = ori_error / np.linalg.norm(ori_error)
        
        return ori_error

    def calc_ori_error_angle(self, goal_ori, current_ori):
        """
        Calculate angle from one rotation to another
        """

        q_delta = current_ori[3]*goal_ori[3] + current_ori[0]*goal_ori[0] + current_ori[1]*goal_ori[1] + current_ori[2]*goal_ori[2] 
        x_delta = current_ori[3]*goal_ori[0] - current_ori[0]*goal_ori[3] + current_ori[1]*goal_ori[2] - current_ori[2]*goal_ori[1]
        y_delta = current_ori[3]*goal_ori[1] - current_ori[1]*goal_ori[3] + current_ori[2]*goal_ori[0] - current_ori[0]*goal_ori[2]
        z_delta = current_ori[3]*goal_ori[2] - current_ori[2]*goal_ori[3] + current_ori[0]*goal_ori[1] - current_ori[1]*goal_ori[0]

        direc2 = np.array([x_delta, y_delta, z_delta])

        angle_error   = 2.0 * np.arctan2(np.linalg.norm(direc2), q_delta)
        angle_error = normalize_angle(angle_error)
        angle_error = np.abs(angle_error)

        #-- Using Rotation matrices
        # Rot_matrix_current = quaternion_matrix(current_ori)
        # Rot_matrix_goal    = quaternion_matrix(goal_ori)

        # Rot_matrix_error   = np.matmul(Rot_matrix_goal, np.transpose(Rot_matrix_current))

        # angle_error, direc, point = rotation_from_matrix(Rot_matrix_error)
        # angle_error = np.abs(angle_error)

        return angle_error


    def get_randomValidGoal(self):
        is_valid = False
        while is_valid is False:
            random_goal = self.get_randomPose()

            random_goal_pos= np.array([random_goal.pose.position.x, random_goal.pose.position.y, random_goal.pose.position.z])
            random_goal_ori= np.array([random_goal.pose.orientation.x, random_goal.pose.orientation.y, random_goal.pose.orientation.z, random_goal.pose.orientation.w])
            is_valid = self.test_goalPose(random_goal_pos, random_goal_ori)
        
        return random_goal_pos, random_goal_ori

    def test_goalPose(self, r_goal_pos, r_goal_ori):
        """
        Function used to check if the defined goal is reachable
        """
        rospy.logwarn("Goal to check: " + str(r_goal_pos) + " " + str(r_goal_ori))
        result = self.check_pose(r_goal_pos, r_goal_ori)
        if result == False:
            rospy.logwarn( "The goal is not reachable")
        
        return result

    