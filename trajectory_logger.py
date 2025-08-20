import os
import csv
import imageio
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from PIL import Image
import torch


class TrajectoryLogger:
    """
    简化的轨迹数据记录器，用于保存PPO训练过程中的轨迹数据
    """
    
    def __init__(self, base_dir: str = "logs"):
        """
        初始化轨迹记录器
        
        Args:
            base_dir (str): 基础日志目录
        """
        self.base_dir = base_dir
        self.log_dir = None
        self.date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_log_directory(self) -> str:
        """
        创建日志目录，每次运行都创建一个新的文件夹
        
        Returns:
            str: 创建的日志目录路径
        """
        self.log_dir = os.path.join(self.base_dir, f"run_{self.date_time}")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"轨迹数据将保存到: {self.log_dir}")
        return self.log_dir
    
    def save_trajectories(self, iteration: int, rewards: torch.Tensor, success: np.ndarray, 
                         values: torch.Tensor, logprobs: torch.Tensor, advantages: torch.Tensor, 
                         returns: torch.Tensor, trajectory_images: List[List[Image.Image]]):
        """
        保存所有环境的轨迹数据（视频+CSV格式）
        
        Args:
            iteration (int): 迭代次数
            rewards (torch.Tensor): 奖励 [num_steps, num_envs]
            success (np.ndarray): 成功状态 [num_steps, num_envs]
            values (torch.Tensor): 状态值 [num_steps, num_envs]
            logprobs (torch.Tensor): 对数概率 [num_steps, num_envs]
            advantages (torch.Tensor): 优势值 [num_steps, num_envs]
            returns (torch.Tensor): 回报值 [num_steps, num_envs]
            trajectory_images (List[List[Image.Image]]): 每个环境的图像序列 [num_envs][num_steps]
        """
        if self.log_dir is None:
            raise ValueError("Log directory has not been created. Call create_log_directory() first.")
        
        num_steps, num_envs = rewards.shape
        
        print("保存轨迹数据...")
        for env_idx in range(num_envs):
            trajectory_data = []
            
            for step in range(num_steps):
                # 构建step数据
                step_data = {
                    'step': step,
                    'reward': rewards[step, env_idx].item(),
                    'success': success[step, env_idx],
                    'value': values[step, env_idx].item(),
                    'logprob': logprobs[step, env_idx].item(),
                    'advantage': advantages[step, env_idx].item(),
                    'return': returns[step, env_idx].item(),
                }
                trajectory_data.append(step_data)
            
            # 确定最终成功状态
            final_success = bool(np.any(success[:, env_idx]))
            
            # 生成文件名
            filepath = self._get_trajectory_filename(iteration, env_idx, final_success)
            
            # 保存CSV
            self._save_csv(trajectory_data, filepath)
            
            # 保存视频
            if env_idx < len(trajectory_images) and trajectory_images[env_idx]:
                self._save_video(trajectory_images[env_idx], filepath)
            
            print(f"环境 {env_idx} 轨迹数据已保存（最终状态: {'成功' if final_success else '失败'}）")
    
    def _get_trajectory_filename(self, iteration: int, env_idx: int, success: bool) -> str:
        """
        生成轨迹文件名
        
        Args:
            iteration (int): 第几次迭代
            env_idx (int): 环境索引
            success (bool): 是否成功
            
        Returns:
            str: 文件路径（不包含扩展名）
        """
        if self.log_dir is None:
            raise ValueError("Log directory has not been created. Call create_log_directory() first.")
        
        success_str = "success" if success else "fail"
        filename = f"iteration_{iteration:04d}_env_{env_idx}__{success_str}"
        return os.path.join(self.log_dir, filename)
    
    def _save_video(self, images: List[Image.Image], filepath: str, fps: int = 30) -> str:
        """
        将图像序列保存为视频文件
        
        Args:
            images (List[Image.Image]): PIL Image对象的列表
            filepath (str): 保存的视频文件路径（不包含扩展名）
            fps (int): 视频帧率
            
        Returns:
            str: 保存的视频文件路径
        """
        if not images:
            return ""
        
        video_path = f"{filepath}.mp4"
        
        # 使用imageio保存视频，参考eval.py中的实现
        video_writer = imageio.get_writer(video_path, fps=fps)
        
        for img in images:
            if isinstance(img, Image.Image):
                # 转换PIL Image为numpy数组
                frame = np.array(img)
            else:
                frame = img
            video_writer.append_data(frame)
        
        video_writer.close()
        print(f"视频已保存到: {video_path}")
        return video_path
    
    def _save_csv(self, trajectory_data: List[Dict[str, Any]], filepath: str) -> str:
        """
        将轨迹数据保存为CSV文件
        
        Args:
            trajectory_data (List[Dict[str, Any]]): 轨迹数据列表，每个元素是一个字典
            filepath (str): 保存的CSV文件路径（不包含扩展名）
            
        Returns:
            str: 保存的CSV文件路径
        """
        if not trajectory_data:
            return ""
        
        csv_path = f"{filepath}.csv"
        
        # 获取所有字段名
        fieldnames = set()
        for step_data in trajectory_data:
            fieldnames.update(step_data.keys())
        fieldnames = sorted(list(fieldnames))
        
        # 写入CSV文件
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trajectory_data)
        
        print(f"轨迹数据已保存到: {csv_path}")
        return csv_path
