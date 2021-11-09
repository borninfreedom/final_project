from typing import Any, List, Tuple, Union, Optional
from easy_logx.easy_logx import EasyLog
import pybullet as p
import pybullet_data
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from math import sqrt
import random
import time
import math
import cv2
import torch
import os
import logging
import pandas as pd
from os.path import exists

pd.set_option("display.max_rows", None, "display.max_columns", None)
logger = EasyLog(log_level=logging.ERROR)
logger.add_filehandler(mode='a')

sdf_loc = './sdf/'

JOINT_INFO_TYPE = 0
JOINT_STATE_TYPE = 1
LINK_STATE_TYPE = 2


# change log
# dv=0.1 in _step func


class SlideBars():
    def __init__(self, Id):
        self.Id = Id
        self.motorNames = []
        self.motorIndices = []
        self.motorLowerLimits = []
        self.motorUpperLimits = []
        self.slideIds = []

        self.numJoints = p.getNumJoints(self.Id)

    def get_motor_indices(self) -> List:
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.Id, i)
            jointName = jointInfo[1].decode('ascii')
            qIndex = jointInfo[3]
            lowerLimits = jointInfo[8]
            upperLimits = jointInfo[9]
            if qIndex > -1:
                self.motorNames.append(jointName)
                self.motorIndices.append(i)
                self.motorLowerLimits.append(lowerLimits)
                self.motorUpperLimits.append(upperLimits)

        return self.motorIndices

    def add_slidebars(self, init_pos: Optional[List] = None) -> None:

        if not self.motorIndices:
            raise "Please run get_motor_indices first."
        for i in range(len(self.motorIndices)):
            if self.motorLowerLimits[i] <= self.motorUpperLimits[i]:
                if not init_pos:
                    slideId = p.addUserDebugParameter(self.motorNames[i], self.motorLowerLimits[i],
                                                      self.motorUpperLimits[i], 0)
                else:
                    slideId = p.addUserDebugParameter(self.motorNames[i], self.motorLowerLimits[i],
                                                      self.motorUpperLimits[i], init_pos[i])
            else:
                if not init_pos:
                    slideId = p.addUserDebugParameter(self.motorNames[i], self.motorUpperLimits[i],
                                                      self.motorLowerLimits[i], 0)
                else:
                    slideId = p.addUserDebugParameter(self.motorNames[i], self.motorUpperLimits[i],
                                                      self.motorLowerLimits[i], init_pos[i])
            self.slideIds.append(slideId)

    def get_slidebars_values(self) -> List:
        slidesValues = []
        for i in self.slideIds:
            value = p.readUserDebugParameter(i)
            slidesValues.append(value)
        return slidesValues


class KukaVisualGraspEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    kMaxEpisodeSteps = 700
    kImageSize = {'width': 96, 'height': 96}
    kFinalImageSize = {'width': 84, 'height': 84}

    def __init__(self, is_render: bool = False, is_good_view: bool = False, skip: int = 4) -> None:

        self.skip = skip
        self.is_render = is_render
        self.is_good_view = is_good_view

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.x_low_obs = 0.35
        self.x_high_obs = 0.85
        self.y_low_obs = -0.3
        self.y_high_obs = 0.3
        self.z_low_obs = 0
        self.z_high_obs = 0.55

        self.x_low_action = -0.5
        self.x_high_action = 0.5
        self.y_low_action = -0.6
        self.y_high_action = 0.6
        self.z_low_action = -0.6
        self.z_high_action = 0.6
        self.theta_low_action = -1.57
        self.theta_high_action = 1.57
        self.step_counter = 0

        self.urdf_root_path = pybullet_data.getDataPath()
        # lower limits for null space
        self.lower_limits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.upper_limits = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.joint_ranges = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.joint_damping = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]

        self.init_joint_positions = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684,
            0, 0, 0, 0, 0, 0
        ]

        self.orientation = p.getQuaternionFromEuler(
            [0., -math.pi, 0])

        self.camera_parameters = {
            'width': 960.,
            'height': 720,
            'fov': 60,
            'near': 0.1,
            'far': 100.,
            'eye_position': [0.59, 0, 0.8],
            'target_position': [0.55, 0, 0.05],
            'camera_up_vector':
                [1, 0, 0],  # I really do not know the parameter's effect.
            'light_direction': [
                0.5, 0, 1
            ],  # the direction is from the light source position to the origin of the world frame.
        }

        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.55, 0, 0.05],
            distance=.7,
            yaw=90,
            pitch=-70,
            roll=0,
            upAxisIndex=2)

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_parameters['fov'],
            aspect=self.camera_parameters['width'] /
                   self.camera_parameters['height'],
            nearVal=self.camera_parameters['near'],
            farVal=self.camera_parameters['far'])

        p.configureDebugVisualizer(lightPosition=[5, 0, 5])
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        self.action_space = spaces.Box(low=np.array(
            [self.x_low_action, self.y_low_action, self.z_low_action, self.theta_low_action]),
            high=np.array([
                self.x_high_action,
                self.y_high_action,
                self.z_high_action,
                self.theta_high_action
            ]),
            dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(skip, self.kFinalImageSize['width'], self.kFinalImageSize['height']),
                                            dtype=np.uint8)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.step_counter = 0

        p.resetSimulation()
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated = False
        p.setGravity(0, 0, -15)

        # 这些是周围那些白线，用来观察是否超过了obs的边界
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"),
                   basePosition=[0, 0, -0.65])
        self.kuka_id = p.loadSDF(os.path.join(sdf_loc, "kuka_with_wsg2.sdf"))[0]
        table_uid = p.loadURDF(os.path.join(self.urdf_root_path,
                                            "table/table.urdf"),
                               basePosition=[0.5, 0, -0.65])
        p.changeVisualShape(table_uid, -1, rgbaColor=[1, 1, 1, 1])

        self.motor_indices = SlideBars(self.kuka_id).get_motor_indices()

        for index, value in zip(self.motor_indices, self.init_joint_positions):
            p.resetJointState(self.kuka_id, index, value)

        for _ in range(15):
            random_urdf_min_index = 0
            random_urdf_max_index = 999
            digit = str(random.randint(random_urdf_min_index, random_urdf_max_index))
            random_urdf_index = (len(str(random_urdf_max_index)) - len(digit)) * '0' + digit

            self.object_id = p.loadURDF(os.path.join(self.urdf_root_path,
                                                     "random_urdfs/" + random_urdf_index + "/" + random_urdf_index + ".urdf"),
                                        basePosition=[
                                            random.uniform(self.x_low_obs,
                                                           self.x_high_obs),
                                            random.uniform(self.y_low_obs,
                                                           self.y_high_obs), 0.02
                                        ])
            p.stepSimulation()

        for _ in range(700):
            p.stepSimulation()

        self.num_joints = p.getNumJoints(self.kuka_id)

        p.stepSimulation()

        (_, _, px, _,
         _) = p.getCameraImage(width=960,
                               height=960,
                               viewMatrix=self.view_matrix,
                               projectionMatrix=self.projection_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
        self.images = px

        self.object_pos = p.getBasePositionAndOrientation(self.object_id)[0]

        self.images = self.images[:, :, :3]  # the 4th channel is alpha channel, we do not need it.

        return self._process_image(self.images)

    def reset(self):
        state = self._reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return np.squeeze(self.random_crop(states, self.kFinalImageSize['width']))

    def _process_image(self, image):
        """Convert the RGB pic to gray pic and add a channel 1

        Args:
            image ([type]): [description]
        """

        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(image, (self.kImageSize['width'], self.kImageSize['height']))[None, :, :]
            return np.array(image, dtype=np.uint8)
        else:
            return np.zeros((1, self.kImageSize['width'], self.kImageSize['height']), dtype=np.uint8)

    def random_crop(self, imgs: np.ndarray, out: int) -> np.ndarray:
        """
            args:
            imgs: shape (B,C,H,W)
            out: output size (e.g. 84)
        """
        n, c, h, w = imgs.shape
        crop_max = h - out + 1
        w1 = np.random.randint(0, crop_max, n)
        h1 = np.random.randint(0, crop_max, n)
        cropped = np.empty((n, c, out, out), dtype=np.uint8)
        for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
            cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
        return cropped

    def _step(self, action: np.float32):
        dv = 0.05
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        theta = action[3]

        self.current_pos = p.getLinkState(self.kuka_id, 6)[4]
        self.new_robot_pos = [
            self.current_pos[0] + dx, self.current_pos[1] + dy,
            self.current_pos[2] + dz
        ]
        self.robot_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.kuka_id,
            endEffectorLinkIndex=6,
            targetPosition=[
                self.new_robot_pos[0], self.new_robot_pos[1],
                self.new_robot_pos[2]
            ],
            targetOrientation=self.orientation
        )
        for i in range(6):
            p.resetJointState(
                self.kuka_id,
                jointIndex=i,
                targetValue=self.robot_joint_positions[i],
            )
        p.resetJointState(self.kuka_id, 6, theta)
        p.stepSimulation()

        # 在代码开始部分，如果定义了is_good_view，那么机械臂的动作会变慢，方便观察
        if self.is_good_view:
            time.sleep(0.05)

        self.step_counter += 1

        return self._reward()

    def _reward(self):

        # p.enableJointForceTorqueSensor(self.kuka_id, 11, 1)
        # 一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明
        self.robot_state = p.getLinkState(self.kuka_id, 6)[4]

        self.object_state = np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]).astype(
            np.float32)

        # square_dx = (self.robot_state[0] - self.object_state[0]) ** 2
        # square_dy = (self.robot_state[1] - self.object_state[1]) ** 2
        # square_dz = (self.robot_state[2] - self.object_state[2]) ** 2

        # 用机械臂末端和物体的距离作为奖励函数的依据
        # self.distance = sqrt(square_dx + square_dy + square_dz)
        # print(self.distance)

        x = self.robot_state[0]
        y = self.robot_state[1]
        z = self.robot_state[2]
        self.is_grasped = False
        if z < 0.3:
            for _ in range(10):
                p.setJointMotorControl2(self.kuka_id, 11, p.POSITION_CONTROL, 0.06, 2, force=300)

                p.setJointMotorControl2(self.kuka_id, 13, p.POSITION_CONTROL, 0.06, 2, force=300)

                p.stepSimulation()

            logger.debug(f'gripper_left_pos={p.getJointState(self.kuka_id, 11)[0]}')

            logger.debug(f'gripper_right_pos={p.getJointState(self.kuka_id, 13)[0]}')
            if p.getJointState(self.kuka_id, 11)[0] >= 0.048 and p.getJointState(self.kuka_id, 13)[0] >= 0.048:
                self.is_grasped = False
            else:
                self.is_grasped = True

        # 如果机械比末端超过了obs的空间，也视为done，而且会给予一定的惩罚
        terminated = bool(x < self.x_low_obs - 0.1 or x > self.x_high_obs + 0.1
                          or y < self.y_low_obs - 0.1 or y > self.y_high_obs + 0.1
                          or z < self.z_low_obs + 0.25 or z > self.z_high_obs + 0.2)

        if terminated:
            reward = -0.1
            self.terminated = True

        # 如果机械臂一直无所事事，在最大步数还不能接触到物体，也需要给一定的惩罚
        elif self.step_counter > self.kMaxEpisodeSteps:
            reward = -0.1
            self.terminated = True

        elif self.is_grasped:
            reward = 1
            self.terminated = True
        else:
            reward = 0
            self.terminated = False

        (_, _, px, _,
         _) = p.getCameraImage(width=960,
                               height=960,
                               viewMatrix=self.view_matrix,
                               projectionMatrix=self.projection_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
        self.images = px
        self.processed_image = self._process_image(self.images)
        logger.debug(f'reward={reward}')
        logger.debug(f'is_grasped={self.is_grasped}')
        return self.processed_image, reward, self.terminated, {}

    def step(self, action: np.float32) -> Tuple[np.ndarray, float, bool, Any]:
        logger.debug(f'action={action}')
        total_reward = 0
        states = []
        state, reward, done, info = self._step(action)
        for i in range(self.skip):
            if not done:
                state, reward, done, info = self._step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        # logger.info(f'total_reward={total_reward}')
        return np.squeeze(self.random_crop(states, self.kFinalImageSize['width'])), total_reward, done, {}

    def close(self):
        p.disconnect()


class ValidateEnv:

    def __init__(self):
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        self.urdf_root_path = pybullet_data.getDataPath()

        self.x_low_obs = 0.3
        self.x_high_obs = 0.8
        self.y_low_obs = -0.3
        self.y_high_obs = 0.3
        self.z_low_obs = 0
        self.z_high_obs = 0.55

        self.x_low_action = -0.4
        self.x_high_action = 0.4
        self.y_low_action = -0.4
        self.y_high_action = 0.4
        self.z_low_action = -0.6
        self.z_high_action = 0.3

        # lower limits for null space
        self.lower_limits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.upper_limits = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.joint_ranges = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.joint_damping = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]

        self.init_joint_positions = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684,
            0, 0, 0, 0, 0, 0
        ]

        self.orientation = p.getQuaternionFromEuler(
            [0., -math.pi, 0])

        p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"),
                   basePosition=[0, 0, -0.65])
        self.kuka_id = p.loadSDF(os.path.join(sdf_loc, "kuka_with_wsg2.sdf"))[0]

        table_uid = p.loadURDF(os.path.join(self.urdf_root_path,
                                            "table/table.urdf"),
                               basePosition=[0.5, 0, -0.65])

        # self.object_id = p.loadURDF(os.path.join(self.urdf_root_path, "random_urdfs/000/000.urdf"),
        #                             basePosition=[0.5, 0, 0])
        p.changeVisualShape(table_uid, -1, rgbaColor=[1, 1, 1, 1])

        for _ in range(10):
            random_urdf_min_index = 0
            random_urdf_max_index = 999
            digit = str(random.randint(random_urdf_min_index, random_urdf_max_index))
            random_urdf_index = (len(str(random_urdf_max_index)) - len(digit)) * '0' + digit

            # self.object_id = p.loadURDF(os.path.join(self.urdf_root_path,
            #                                          "random_urdfs/" + random_urdf_index + "/" + random_urdf_index + ".urdf"),
            #                             basePosition=[
            #                                 random.uniform(self.x_low_obs,
            #                                                self.x_high_obs),
            #                                 random.uniform(self.y_low_obs,
            #                                                self.y_high_obs), 0.01
            #                             ])

        p.configureDebugVisualizer(lightPosition=[5, 0, 5])
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        logger.info(f'motor indices={SlideBars(self.kuka_id).get_motor_indices()}')
        self.slide_bars = SlideBars(self.kuka_id)
        self.motor_indices = self.slide_bars.get_motor_indices()
        self.slide_bars.add_slidebars(self.init_joint_positions)

        self.num_joints = p.getNumJoints(self.kuka_id)

        self.get_joints_info()

        for index, value in zip(self.motor_indices, self.init_joint_positions):
            p.resetJointState(self.kuka_id, index, value)

    def slidebar_debug(self):
        while True:
            p.stepSimulation()
            slide_values = self.slide_bars.get_slidebars_values()
            p.setJointMotorControlArray(self.kuka_id, self.motor_indices, p.POSITION_CONTROL, slide_values)
            # self.get_joint_state(6)
            # self.get_joint_state(11)
            # self.get_joint_state(12)

    def get_joints_info(self):
        joint_infos = []
        for i in range(self.num_joints):
            joint_infos.append(p.getJointInfo(self.kuka_id, i))

        joint_info_df = pd.DataFrame(joint_infos,
                                     columns=['index', 'name', 'type', 'qindex', 'uindex', 'flags', 'dampimg',
                                              'friction', 'lower limit', 'upper limit', 'max force', 'max vel',
                                              'link name', 'axis', 'parent frame pos', 'parent frame orn',
                                              'parent index'], dtype=object)

        logger.debug('joints infos = ')
        logger.debug(joint_info_df)

    def get_joint_state(self, index: int) -> None:
        joint_state = [p.getJointState(self.kuka_id, index)]
        # logger.debug(f'joint_state={joint_state}')
        joint_state_df = pd.DataFrame(joint_state,
                                      columns=['pos', 'vel', 'force', 'torque'],
                                      dtype=object)
        logger.debug(f'joint {index} state = ')
        logger.debug(joint_state_df)

    def get_link_state(self, index: int) -> None:
        link_state = p.getLinkState(self.kuka_id, index)
        link_state_df = pd.DataFrame(link_state,
                                     columns=['world pos', 'world orn', 'local inertial frame pos',
                                              'local inertial frame orn', 'world frame pos', 'world frame orn',
                                              'linear vel', 'angular vel'], dtype=object)
        logger.debug(f'link {index} state = ')
        logger.debug(link_state_df)

    def sim_grasp(self):

        # for index, value in zip(self.motor_indices, self.init_joint_positions):
        #     p.resetJointState(self.kuka_id, index, value)
        target_pos = [0.5, 0, 0.29]
        init_pos = self.init_joint_positions[:7]

        robot_joint_positions = p.calculateInverseKinematics(
            self.kuka_id,
            6,
            target_pos,
            self.orientation,
            self.joint_damping,
        )
        for i in range(7):
            p.resetJointState(
                self.kuka_id,
                jointIndex=i,
                targetValue=robot_joint_positions[i],
            )

        # p.enableJointForceTorqueSensor(self.kuka_id, 11, 1)

        force_sensor = p.getJointState(self.kuka_id, 11)[2][1]
        while True:
            end_pos = p.getLinkState(self.kuka_id, 6)[4][2]
            if end_pos < 0.3:
                # for _ in range(3):
                #     p.setJointMotorControl2(self.kuka_id, 11, p.POSITION_CONTROL, 0.045, 2, force=300)
                #
                #     p.setJointMotorControl2(self.kuka_id, 13, p.POSITION_CONTROL, 0.045, 2, force=300)
                #     p.stepSimulation()
                #     time.sleep(0.1)

                while abs(p.getJointState(self.kuka_id, 11)[2][1]) < 80:
                    # p.resetJointState(self.kuka_id,11,0.04)
                    # p.resetJointState(self.kuka_id,13,0.04)

                    # p.setJointMotorControlArray(self.kuka_id,[11,13],p.POSITION_CONTROL,[0.045,0.045],forces=[200,200])
                    p.setJointMotorControl2(self.kuka_id, 11, p.POSITION_CONTROL, 0.06, 2, force=300)

                    p.setJointMotorControl2(self.kuka_id, 13, p.POSITION_CONTROL, 0.06, 2, force=300)

                    # logger.debug(f'debug_force={p.getJointState(self.kuka_id, 11)[2]}')
                    p.stepSimulation()
                    time.sleep(0.1)
                    logger.debug(f'gripper_left_state={p.getJointState(self.kuka_id, 11)}')

                    logger.debug(f'gripper_right_state={p.getJointState(self.kuka_id, 13)}')

            # logger.debug(f'force sensor={p.getJointState(self.kuka_id, 11)[2]}')
            # p.stepSimulation()

            logger.debug(f'debug_force={p.getJointState(self.kuka_id, 11)[2]}')
            time.sleep(0.1)
            for i in range(7):
                p.setJointMotorControlArray(self.kuka_id, self.motor_indices[:7], p.POSITION_CONTROL,
                                            self.init_joint_positions[:7], [2] * 7)

            # logger.debug(f'object pos={p.getBasePositionAndOrientation(self.object_id)[0][2]}')
            # slide_values = self.slide_bars.get_slidebars_values()
            # p.setJointMotorControlArray(self.kuka_id, self.motor_indices, p.POSITION_CONTROL, slide_values)
            p.stepSimulation()


if __name__ == '__main__':
    env = KukaVisualGraspEnv(is_render=True)
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        obs_, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
    # env = ValidateEnv()
    # # env.slidebar_debug()
    # env.sim_grasp()
