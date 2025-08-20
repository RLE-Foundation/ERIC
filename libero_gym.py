"""
LIBERO环境的Gym包装器
提供与Gym兼容的LIBERO机器人学习环境
"""

import os
import numpy as np
import tensorflow as tf
import gymnasium
import gymnasium as gym
import gymnasium.spaces
import gymnasium.vector
from gymnasium.envs import registration
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


class LiberoEnv(gym.Env):
    def __init__(self, task_suite_name, task_id, max_steps, init_state_id, seed, height=128, width=128):
        """
        Args:
            task_suite_name (str): Name of the task suite
            task_id (int): ID of the task
            seed (int): Random seed
            max_steps (int): Maximum number of steps per episode
            init_state_id (int): ID of the initial state

        Returns:
            A LIBERO environment.
        """
        # Set the task information
        self.task_id = task_id
        self.task_suite_name = task_suite_name
        self.max_steps = max_steps
        self.init_state_id = init_state_id
        self.seed = seed
        self.height = height
        self.width = width
        
        # Create the environment
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        task = task_suite.get_task(task_id)
        self.task_name = task.name
        self.task_description = task.language
        self.prompt = f"In: What action should the robot take to {self.task_description.lower()}?\nOut:"
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        
        self.env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=self.height, camera_widths=self.width)

        # Get the initial states
        self.initial_states = task_suite.get_task_init_states(task_id)
        self.init_state = self.initial_states[init_state_id]

        # Set a step counter
        self.current_step = 0
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        # 注意：这里的observation_space是处理后的图像空间，实际返回的是PIL图像
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
        
    def step(self, action):
        action = self.calibrate_action(action)
        obs, reward, done, info = self.env.step(action)
        visual_obs = self.get_visual_obs(obs)

        # Update the current step
        self.current_step += 1

        truncated = self.current_step >= self.max_steps
        
        # Update the info
        info["prompt"] = self.prompt
        info["success"] = done

        return visual_obs, reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        if seed:
            self.seed = seed
         
        # Reset the environment
        self.env.seed(self.seed)
        self.env.reset()
        # Set the initial state
        self.env.set_state(self.init_state)
        
        # Run 10 dummy steps to wait for the environment to stabilize
        dummy_action = [0.] * 7
        obs, reward, done, info = None, None, None, {}
        for _ in range(10):
            obs, reward, done, info = self.env.step(dummy_action)
        
        # Get the visual observation
        visual_obs = self.get_visual_obs(obs)

        # Update the info
        info['prompt'] = self.prompt
        info['success'] = done

        return visual_obs, info

    def close(self):
        self.env.close()
    
    def calibrate_action(self, action, binarize=True):
        """
        1. Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
        Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
        Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
        the dataset wrapper. Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

        2. Flips the sign of the gripper action (last dimension of action vector).
        This is necessary for some environments where -1 = open, +1 = close, since
        the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

        Args:
            action (np.ndarray): Action vector.
            binarize (bool): Whether to binarize the gripper action.

        Returns:
            np.ndarray: Calibrated action vector.
        """
        # Create a copy of the input array to make it writable
        action = np.array(action, copy=True)
        
        # Just normalize the last action to [-1,+1].
        orig_low, orig_high = 0.0, 1.0
        action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

        if binarize:
            # Binarize to -1 or +1.
            action[..., -1] = np.sign(action[..., -1])
        
        # Invert the gripper action
        action[..., -1] = action[..., -1] * -1.0
        return action
    
    def get_visual_obs(self, obs, resize_size=None):
        """
        Get the visual observation from the environment.

        Args:
            obs (dict): Observation from the environment.
            resize_size (tuple): Resize size of the image.

        Returns:
            np.ndarray: Visual observation as numpy array with shape (224, 224, 3).
        """
        agentview_image = obs['agentview_image']
        # NOTE: rotate 180 degrees to match train preprocessing
        agentview_image = agentview_image[::-1, ::-1]
        # Resize to image size expected by model
        if resize_size is None:
            resize_size = (self.height, self.width)
        assert isinstance(resize_size, tuple)
        img = tf.image.encode_jpeg(agentview_image)
        img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
        img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
        img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
        
        # 返回numpy数组而不是PIL Image，以便与observation_space一致
        return img.numpy().astype(np.uint8)
    
    def get_visual_obs_as_pil(self, obs, resize_size=None):
        """
        Get the visual observation as PIL Image (for compatibility with existing code).

        Args:
            obs (dict): Observation from the environment.
            resize_size (tuple): Resize size of the image.

        Returns:
            PIL.Image: Visual observation as PIL Image.
        """
        if resize_size is None:
            resize_size = (self.height, self.width)
        img_array = self.get_visual_obs(obs, resize_size)
        img = Image.fromarray(img_array)
        img = img.convert("RGB")
        return img


def register_libero_env(task_suite_name='libero_spatial', task_id=0, max_steps=250, init_state_id=0, seed=42, height=128, width=128):
    """注册LIBERO环境到gym环境注册表
    
    Args:
        task_suite_name (str): 任务套件名称
        task_id (int): 任务ID  
        max_steps (int): 每个episode的最大步数
        init_state_id (int): 初始状态ID
        seed (int): 随机种子
        
    Returns:
        str: 注册的环境ID
    """
    env_id = f'LiberoSpatial-{task_id}-v0'
    if env_id in registration.registry:
        print(f"{env_id}环境已存在，跳过注册")
        return env_id
     
    try:
        gym.register(
            id=env_id,
            entry_point='libero_gym:LiberoEnv',
            kwargs={
                'task_suite_name': task_suite_name,
                'task_id': task_id,
                'max_steps': max_steps,
                'init_state_id': init_state_id,
                'seed': seed,
                'height': height,
                'width': width
            },
            nondeterministic=True
        )
        print(f"成功注册{env_id}环境")
        return env_id
    except Exception as e:
        print(f"注册环境时出错: {e}")
        return env_id


def make_libero_vec_env(task_suite_name='libero_spatial', task_id=0, max_steps=250, 
                       init_state_id=0, seed=42, num_envs=1, height=128, width=128) -> gymnasium.vector.VectorEnv:
    """创建LIBERO向量化环境
    
    Args:
        task_suite_name (str): 任务套件名称
        task_id (int): 任务ID
        max_steps (int): 每个episode的最大步数
        init_state_id (int): 初始状态ID
        seed (int): 随机种子
        num_envs (int): 环境数量
        
    Returns:
        gym.vector.VectorEnv: 向量化环境
    """
    # 注册环境
    env_id = register_libero_env(
        task_suite_name=task_suite_name,
        task_id=task_id,
        max_steps=max_steps,
        init_state_id=init_state_id,
        seed=seed,
        height=height,
        width=width
    )
    
    # 创建向量化环境
    # 使用 SyncVectorEnv 手动创建多个环境实例
    env_fns = [lambda: gym.make(env_id) for _ in range(num_envs)]
    envs = gymnasium.vector.SyncVectorEnv(env_fns)
    print(f"成功创建了{num_envs}个向量化环境")
    
    return envs
