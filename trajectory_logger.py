import csv
import imageio
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
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
        self.base_dir = Path(base_dir)
        self.log_dir = None
        self.date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_log_directory(self) -> Path:
        """
        创建或设置日志目录
        如果 base_dir 已经是完整的运行目录（包含时间戳），则直接使用
        否则创建一个新的时间戳子目录
        
        Returns:
            Path: 创建的日志目录路径
        """
        self.log_dir = self.base_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"轨迹数据将保存到: {self.log_dir}")
        return self.log_dir
    
    def save_trajectories(self, iteration: int, rewards: torch.Tensor, success: np.ndarray, 
                         values: torch.Tensor, logprobs: torch.Tensor, advantages: torch.Tensor, 
                         returns: torch.Tensor, dones: torch.Tensor, trajectory_images: List[List[Image.Image]]):
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
            dones (torch.Tensor): 完成状态 [num_steps, num_envs]
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
                    'done': dones[step, env_idx].item(),
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
            
            # 保存视频，添加文字叠加
            if env_idx < len(trajectory_images) and trajectory_images[env_idx]:
                self._save_video(trajectory_images[env_idx], filepath, add_text=True, 
                               trajectory_data=trajectory_data)
            
            print(f"环境 {env_idx} 轨迹数据已保存（最终状态: {'成功' if final_success else '失败'}）")
    
    def _get_trajectory_filename(self, iteration: int, env_idx: int, success: bool) -> Path:
        """
        生成轨迹文件名
        
        Args:
            iteration (int): 第几次迭代
            env_idx (int): 环境索引
            success (bool): 是否成功
            
        Returns:
            Path: 文件路径（不包含扩展名）
        """
        if self.log_dir is None:
            raise ValueError("Log directory has not been created. Call create_log_directory() first.")
        
        success_str = "success" if success else "fail"
        filename = f"iteration_{iteration:04d}_env_{env_idx}__{success_str}"
        return self.log_dir / filename
    
    def _save_video(self, images: List[Image.Image], filepath: Path, fps: int = 30, 
                   add_text: bool = False, trajectory_data: Optional[List[Dict[str, Any]]] = None) -> Path:
        """
        将图像序列保存为视频文件
        
        Args:
            images (List[Image.Image]): PIL Image对象的列表
            filepath (Path): 保存的视频文件路径（不包含扩展名）
            fps (int): 视频帧率
            add_text (bool): 是否在视频帧上添加文字信息
            trajectory_data (List[Dict[str, Any]]): 轨迹数据，用于添加文字信息
            
        Returns:
            Path: 保存的视频文件路径
        """
        if not images:
            return Path()
        
        video_path = filepath.with_suffix('.mp4')
        
        # 使用imageio保存视频，参考eval.py中的实现
        video_writer = imageio.get_writer(str(video_path), fps=fps)
        
        for idx, img in enumerate(images):
            if isinstance(img, Image.Image):
                frame_img = img.copy()
            else:
                frame_img = Image.fromarray(img)
            
            # 如果需要添加文字并且有轨迹数据
            if add_text and trajectory_data and idx < len(trajectory_data):
                frame_img = self._add_text_to_frame(frame_img, trajectory_data[idx])
            
            # 转换PIL Image为numpy数组
            frame = np.array(frame_img)
            video_writer.append_data(frame)
        
        video_writer.close()
        print(f"视频已保存到: {video_path}")
        return video_path
    
    def _add_text_to_frame(self, image: Image.Image, step_data: Dict[str, Any]) -> Image.Image:
        """
        在图像帧上添加文字信息
        
        Args:
            image (Image.Image): 原始图像
            step_data (Dict[str, Any]): 包含step、reward、done、success信息的字典
            
        Returns:
            Image.Image: 添加了文字的图像
        """
        # 创建图像副本避免修改原图
        img_with_text = image.copy()
        draw = ImageDraw.Draw(img_with_text)
        
        # 尝试使用默认字体，如果失败则使用系统默认字体
        try:
            # 使用较小的字体大小
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 12)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except:
                # 如果找不到字体文件，使用默认字体
                font = ImageFont.load_default()
        
        # 准备要显示的文字信息
        step = step_data.get('step', 0)
        reward = step_data.get('reward', 0.0)
        done = step_data.get('done', 0)
        success = step_data.get('success', 0)
        
        # 格式化文字，保持简洁
        text_lines = [
            f"Step: {step}",
            f"Reward: {reward:.3f}",
            f"Done: {done}",
            f"Success: {success}"
        ]
        
        # 在左上角绘制白色文字，背景为半透明黑色
        margin = 5
        line_height = 15
        
        for i, line in enumerate(text_lines):
            y_position = margin + i * line_height
            
            # 获取文字大小来绘制背景矩形
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 绘制半透明黑色背景
            draw.rectangle([margin - 2, y_position - 2, 
                           margin + text_width + 2, y_position + text_height + 2], 
                          fill=(0, 0, 0, 128))
            
            # 绘制白色文字
            draw.text((margin, y_position), line, fill=(255, 255, 255), font=font)
        
        return img_with_text
    
    def _save_csv(self, trajectory_data: List[Dict[str, Any]], filepath: Path) -> Path:
        """
        将轨迹数据保存为CSV文件
        
        Args:
            trajectory_data (List[Dict[str, Any]]): 轨迹数据列表，每个元素是一个字典
            filepath (Path): 保存的CSV文件路径（不包含扩展名）
            
        Returns:
            Path: 保存的CSV文件路径
        """
        if not trajectory_data:
            return Path()
        
        csv_path = filepath.with_suffix('.csv')
        
        # 获取所有字段名
        fieldnames = set()
        for step_data in trajectory_data:
            fieldnames.update(step_data.keys())
        fieldnames = sorted(list(fieldnames))
        
        # 写入CSV文件
        with csv_path.open('w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trajectory_data)
        
        print(f"轨迹数据已保存到: {csv_path}")
        return csv_path
