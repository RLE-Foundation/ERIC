# Import necessary libraries
# Basic Python libraries for various operations
import random
import tyro
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CURL_CA_BUNDLE'] = ''
import numpy as np
import torch
import torch.distributed as dist

import gymnasium as gym
import time
import json
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm

from accelerate import PartialState
from dataclasses import dataclass
from peft import LoraConfig, get_peft_model
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import AutoConfig, AutoImageProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from libero_gym import make_libero_vec_env
from PIL import Image
from trajectory_logger import TrajectoryLogger


def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None

    Explanation:
        1. Sets seed for Python's built-in random module for basic random operations.
        2. Sets seed for NumPy, ensuring consistent random number generation in array operations.
        3. Sets seed for PyTorch CPU operations.
        4. If CUDA is available, sets seed for all GPU devices.
        5. Configures cuDNN to ensure deterministic behavior:
           - Sets deterministic flag to True, ensuring reproducible results.
           - Disables benchmarking to prevent algorithm selection based on hardware.

    Note:
        Setting deterministic behavior may impact performance but ensures consistent results
        across multiple runs, which is crucial for debugging and research.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging_system(args) -> tuple[str, Path]:
    """
    设置统一的日志系统，创建日志目录并保存参数
    
    Args:
        args: 命令行参数对象
        
    Returns:
        tuple[str, Path]: (run_name, log_dir) 运行名称和日志目录路径
    """
    # 生成运行名称，使用时间字符串作为后缀
    run_name = f"{args.exp_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # 创建日志目录
    log_dir = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存参数到JSON文件以便复现
    args_dict = vars(args)
    args_file = log_dir / "args.json"
    with args_file.open('w', encoding='utf-8') as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"日志系统初始化完成:")
    print(f"  运行名称: {run_name}")
    print(f"  日志目录: {log_dir}")
    print(f"  参数文件: {args_file}")
    
    return run_name, log_dir

# Call the function to set random seed for reproducibility
# set_random_seed(42)

# Set environment variables for Weights & Biases (wandb) logging
os.environ["WANDB_API_KEY"] = "USE YOUR KEY"
os.environ["WANDB_PROJECT"] = "VLA_with_RL_Finetuning_From_Scratch"

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanVLA-RL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    debug: bool = False
    """whether to use debugpy"""
    save_log: bool = False
    """whether to save trajectory data (videos and CSV files)"""

    # VLA arguments
    vla_model_path: str = "/home/zixiao/models/openvla-7b-finetuned-libero-spatial"
    """the path to the VLA model"""
    lora_rank: int = 32
    """the rank of the LoRA matrix"""
    lora_dropout: float = 0.0
    """the dropout rate of the LoRA matrix"""
    max_seq_len: int = 128
    """the maximum sequence length of the input"""

    # Algorithm specific arguments
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments (supports vectorized environments)"""
    num_steps: int = 250
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    minibatch_size: int = 8
    """the size of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

class Agent(nn.Module):
    def __init__(
        self, 
        vla_base, 
        processor,
        device,
        unnorm_key,
        vla_config: OpenVLAConfig
    ):
        super().__init__()

        self.vla_base = vla_base
        self.processor = processor
        self.device = device
        self.config = vla_base.module.config.text_config
        # n_embd 是 transformer 的 hidden size（宽度）
        self.config.n_embd = self.config.hidden_size if hasattr(
            self.config, "hidden_size") else self.config.n_embd

        self.value_head = nn.Sequential(
            nn.Linear(self.config.n_embd, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 1, bias=False),
        )
        
        # for action decoding
        self.vla_config = vla_config
        self.unnorm_key = unnorm_key
        assert self.vla_config.norm_stats is not None, "norm_stats is not found in config"
        self.norm_stats: Dict = self.vla_config.norm_stats
        assert self.unnorm_key in self.norm_stats, f"unnorm_key {self.unnorm_key} not found in norm_stats"
        # Compute action bins
        self.bins = np.linspace(-1, 1, vla_config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.vla_config.text_config.vocab_size - self.vla_config.pad_to_multiple_of
    
    def get_value(self, input_ids, attention_mask, pixel_values, **kwargs,):
        # TODO: value net 的实例问题
        transformer_outputs = self.vla_base(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values,    # [B, 3, H, W]
            output_hidden_states=True,
            **kwargs,
        )

        hidden_states = transformer_outputs.hidden_states[-1] # [B, L, D], e.g. [2, 292, 4096]
        # TODO: 为什么只取最后一个 token 的 hidden state？
        hidden_states = hidden_states[:, -1, :].float()  # [B, D], e.g. [2, 4096]

        values = self.value_head(hidden_states)

        return values

    def process_to_inputs(self, prompt: str, obs_img: Image.Image, seq_len: int, **kwargs):
        """给输入数据编码，给定prompt和obs_img，返回query_inputs

        Args:
            prompt (str): prompt
            obs_img (Image.Image): observation image
            seq_len (int): 目标序列长度，会进行padding或截断到此长度

        Returns:
            dict: query inputs
        """
        # 先不设置max_length，让processor正常处理
        query_inputs = self.processor(prompt, obs_img, **kwargs).to(self.device, dtype=torch.bfloat16)

        input_ids = query_inputs['input_ids']
        attention_mask = query_inputs['attention_mask']
        
        # 追加特殊token 29871（如果需要）
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
            attention_mask = torch.cat(
                (attention_mask, torch.unsqueeze(torch.Tensor([1]).long(), dim=0).to(attention_mask.device)), dim=1
            )
        
        # TODO: padding 会导致输出错误
        # 根据seq_len进行左侧padding或截断
        # current_length = input_ids.shape[1]
        # if current_length < seq_len:
        #     # 需要左侧padding
        #     pad_length = seq_len - current_length
        #     # 使用tokenizer的pad_token_id进行padding，如果没有则使用32000
        #     pad_token_id = getattr(self.processor.tokenizer, 'pad_token_id', 32000)
            
        #     # 左侧padding input_ids
        #     pad_ids = torch.full((input_ids.shape[0], pad_length), pad_token_id, 
        #                        dtype=input_ids.dtype, device=input_ids.device)
        #     input_ids = torch.cat([pad_ids, input_ids], dim=1)
            
        #     # 左侧padding attention_mask (padding部分设为0)
        #     pad_mask = torch.zeros((attention_mask.shape[0], pad_length), 
        #                          dtype=attention_mask.dtype, device=attention_mask.device)
        #     attention_mask = torch.cat([pad_mask, attention_mask], dim=1)
            
        # elif current_length > seq_len:
        #     # 需要截断（从左侧截断，保留右侧的实际内容）
        #     print("Warning: input_ids length exceed max_seq_len, will be truncated!")
        #     input_ids = input_ids[:, -seq_len:]
        #     attention_mask = attention_mask[:, -seq_len:]
        
        query_inputs['input_ids'] = input_ids
        query_inputs['attention_mask'] = attention_mask
        
        return query_inputs

    def act_rollout(self, query_inputs):
        """给定inputs，返回action
        需要获取模型所有输出，不仅是 action，因此不直接调用 predict_action，而是使用 generate 方法
        只支持 batch_size = 1

        Args:
            query_inputs (dict): query inputs

        Returns:
            action (np.ndarray): action
            action_token_ids (np.ndarray): action token ids
            values (torch.Tensor): values
            log_probs (torch.Tensor): log probabilities
        """
        outputs = self.vla_base.module.generate(
            **query_inputs,
            max_new_tokens=7,
            do_sample=True,
            top_p=1.0,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_logits=True,
            output_hidden_states=True,
        )

        action_token_ids = outputs.sequences[:, -7:]
        # shape: [batch_size, seq_len]
        
        # TODO: batch_size > 1 时，可能有问题
        logits = torch.concat(outputs.logits, dim=0).reshape(1, 7, -1)
        # shape: [batch_size, seq_len, vocab_size]
        
        # 一整个 action（7个token） 对应一个 log_probs
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs.gather(dim=-1, index=action_token_ids.unsqueeze(-1)).squeeze(-1)
        log_probs = log_probs.sum(dim=-1).reshape(-1, 1)
        # shape: [batch_size, 1]
        
        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = action_token_ids[0, :].cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        # Unnormalize actions
        action_norm_stats = self.norm_stats[self.unnorm_key]["action"]
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        actions = actions.reshape(1, -1)

        # TODO: 使用第一个还是最后一个token的hidden state？
        hidden_states = outputs.hidden_states[-1][-1][:, -1, :]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values = self.value_head(hidden_states)

        # 构造完整的 attention_mask 和 sequences，用于后续梯度计算
        sequences = outputs.sequences
        attention_mask = torch.cat(
            (
                query_inputs['attention_mask'], 
                torch.ones((1, 7), device=query_inputs['attention_mask'].device, dtype=torch.long)
            ),
            dim=1
        )
        assert sequences.shape[1] == attention_mask.shape[1], f"sequences.shape[1] != attention_mask.shape[1]"
        return actions, values, log_probs, sequences, attention_mask

    def encode_traj(self, sequences, attention_mask, pixel_values):
        """对已经采集的动作重新编码一次，用于计算梯度

        Args:
            sequences (torch.Tensor): shape: [batch_size, max_seq_len] 包含完整的输入序列和action tokens
            attention_mask (torch.Tensor): shape: [batch_size, max_seq_len]
            pixel_values (torch.Tensor): shape: [batch_size, 6, 224, 224]
        
        Returns:
            action_log_probs (torch.Tensor): shape: [batch_size] 动作的对数概率
            values (torch.Tensor): shape: [batch_size] 状态值
        """
        assert sequences.shape[1] == attention_mask.shape[1], f"sequences.shape[1] != attention_mask.shape[1]"
        
        # 直接使用完整的sequences进行前向传播
        outputs = self.vla_base(
            input_ids=sequences,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        batch_size = sequences.shape[0]
        seq_len = sequences.shape[1]
        # 获取logits，去掉开头的图片token
        logits = outputs.logits[:, -seq_len:, :]  # [batch_size, seq_len, vocab_size]
        
        # 找到最后一个mask=1的位置
        batch_indices = torch.arange(batch_size, device=attention_mask.device)
        flipped_mask = torch.flip(attention_mask, dims=[1])
        
        # 输入序列中最后一个 action 的位置
        last_valid_pos = seq_len - 1 - torch.argmax(flipped_mask.float(), dim=1)  # [batch_size]
        # 输入序列中，第一个 action 的位置
        action_start_pos = last_valid_pos - 6  # [batch_size]
        action_positions = action_start_pos.unsqueeze(1) + torch.arange(7, device=attention_mask.device)  # [batch_size, 7]
        
        # 检查最后7个位置的和是否为7（即都是1且连续）
        action_mask_sum = attention_mask[batch_indices.unsqueeze(1), action_positions].sum(dim=1)  # [batch_size]
        invalid_mask = action_mask_sum != 7
        if invalid_mask.any():
            invalid_indices = torch.where(invalid_mask)[0]
            raise ValueError(f"样本 {invalid_indices.tolist()} 的attention_mask最后7个位置不全为1或不连续")
        
        # 向量化提取action logits和token ids
        # 输出序列中，用来预测 action 对应的 logits 位置
        logit_positions = action_positions - 1  # [batch_size, 7]
        action_logits = logits[batch_indices.unsqueeze(1), logit_positions]  # [batch_size, 7, vocab_size]
        action_token_ids = sequences[batch_indices.unsqueeze(1), action_positions]  # [batch_size, 7]
        
        # 计算log probabilities
        log_probs = torch.log_softmax(action_logits, dim=-1)  # [batch_size, 7, vocab_size]
        action_log_probs = log_probs.gather(dim=-1, index=action_token_ids.unsqueeze(-1)).squeeze(-1)  # [batch_size, 7]
        action_log_probs = action_log_probs.sum(dim=-1)  # [batch_size]
        
        # 向量化计算value
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        last_hidden = hidden_states[batch_indices, last_valid_pos]  # [batch_size, hidden_size]
        values = self.value_head(last_hidden.float()).squeeze(-1)  # [batch_size]
        
        return action_log_probs, values


class PPODataset(Dataset):
    """
    PPO训练数据的Dataset类，用于配合DataLoader和DistributedSampler进行分布式训练
    """
    def __init__(self, sequences, attention_mask, pixel_values, logprobs, advantages, returns, values):
        """
        Args:
            sequences (torch.Tensor): [batch_size, max_seq_len] 完整的输入序列和action tokens
            attention_mask (torch.Tensor): [batch_size, max_seq_len] 注意力掩码
            pixel_values (torch.Tensor): [batch_size, 6, 224, 224] 图像数据
            logprobs (torch.Tensor): [batch_size] 动作的对数概率
            advantages (torch.Tensor): [batch_size] 优势值
            returns (torch.Tensor): [batch_size] 回报值
            values (torch.Tensor): [batch_size] 状态值
        """
        self.sequences = sequences
        self.attention_mask = attention_mask
        self.pixel_values = pixel_values
        self.logprobs = logprobs
        self.advantages = advantages
        self.returns = returns
        self.values = values
        
        # 确保所有数据的batch_size一致
        assert len(sequences) == len(attention_mask) == len(pixel_values) == len(logprobs) == len(advantages) == len(returns) == len(values)
        
        # 给每个样本分配连续ID，验证分布式数据采样
        self.sample_ids = torch.arange(len(sequences))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': self.sequences[idx],
            'attention_mask': self.attention_mask[idx],
            'pixel_values': self.pixel_values[idx],
            'logprobs': self.logprobs[idx],
            'advantages': self.advantages[idx],
            'returns': self.returns[idx],
            'values': self.values[idx],
            'sample_id': self.sample_ids[idx]
        }


def main():
    # DDP setup
    assert torch.cuda.is_available(), "CUDA is not available"
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=int(os.environ.get('WORLD_SIZE', 1)),
            rank=int(os.environ.get('RANK', 0))
        )
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{distributed_state.local_process_index}")

    # 参数解析与日志
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    # 设置统一的日志系统
    run_name, log_dir = setup_logging_system(args)
    
    # 验证参数
    if args.num_envs <= 0:
        raise ValueError(f"num_envs必须大于0，当前值：{args.num_envs}")
    print(f"使用{args.num_envs}个并行环境进行训练")
    
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    # 创建tensorboard writer，使用统一的日志目录下的tensorboard子目录
    tensorboard_dir = log_dir / "tensorboard"
    writer = SummaryWriter(str(tensorboard_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # 调试设置 - 只在rank 0进程中启用远程调试
    if int(os.environ.get('LOCAL_RANK', 0)) == 0 and args.debug:
        import debugpy
        debugpy.listen(5678)
        print("等待调试器连接到端口 5678...")
        debugpy.wait_for_client()
        print("调试器已连接!")

    # TRY NOT TO MODIFY: seeding
    # TODO: 要不要使用上面的函数
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # env setup
    # 为了计算方便，用经过 processor 编码后的图像 tensor 作为 observation，该 tensor 的 shape 为 (6, 224, 224)
    observation_space = gym.spaces.Box(low=0, high=255, shape=(6, 224, 224), dtype=np.uint8)
    action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
    envs = None
    if distributed_state.is_main_process:
        # 使用专门的函数创建向量化环境
        envs = make_libero_vec_env(
            task_suite_name='libero_spatial',
            task_id=0,
            max_steps=args.num_steps,
            init_state_id=0,
            seed=args.seed,
            num_envs=args.num_envs,
            height=observation_space.shape[1],
            width=observation_space.shape[2]
        )

    # VLA setup
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(args.vla_model_path, trust_remote_code=True)
    vla_base = AutoModelForVision2Seq.from_pretrained(
        args.vla_model_path, 
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to(f"cuda:{distributed_state.local_process_index}")
    vla_config = vla_base.config

    # LoRA setup
    lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=min(args.lora_rank, 16),
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
    vla_base = get_peft_model(vla_base, lora_config)

    # DDP setup
    vla_base = DDP(vla_base, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # agent setup
    agent = Agent(
        vla_base, 
        processor, 
        device, 
        unnorm_key="libero_spatial",
        vla_config=vla_config
    ).to(device)
    
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    # 收集数据，在所有 GPU 上创建一份，rollout结束后，子进程会从主进程接收广播数据
    # 轨迹数据，用于计算梯度
    traj_obs = torch.zeros((args.num_steps, args.num_envs) + observation_space.shape, dtype=torch.bfloat16).to(device)
    # 包含 action 的完整输出序列
    traj_attention_mask = torch.zeros((args.num_steps, args.num_envs, args.max_seq_len), dtype=torch.long).to(device)
    # 加入 action 对应 mask 的完整 mask
    traj_sequences = torch.zeros((args.num_steps, args.num_envs, args.max_seq_len), dtype=torch.long).to(device)
    
    # RL 算法需要的数据
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    advantages = torch.zeros((args.num_steps, args.num_envs)).to(device)
    returns = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # 用于记录指标，action 后是否成功
    success = np.zeros((args.num_steps, args.num_envs), dtype=np.int8)
    trajectory_images = None
    trajectory_logger = None
    if args.save_log and distributed_state.is_main_process:
        # 使用统一的日志目录作为 TrajectoryLogger 的基础目录
        trajectory_logger = TrajectoryLogger(base_dir=str(log_dir / "trajectory"))
        trajectory_logger.create_log_directory()

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    # 大循环，内部先 rollout 再更新
    for iteration in range(1, args.num_iterations + 1):
        print("================================================================================")
        print(f"iteration {iteration}/{args.num_iterations}")
        print("================================================================================")
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # ==================================================================================================================
        # Stage 1, 2: Rollout, Compute Data
        # ==================================================================================================================
        # NOTE: only the main process will do the action
        if distributed_state.is_main_process:
            assert envs is not None
            
            # Reset environment at the beginning of each iteration to start a new episode
            print("Resetting environment for new episode...")
            obs_imgs, infos = envs.reset()
            next_done = torch.zeros(args.num_envs).to(device)
            traj_attention_mask.zero_()
            
            if args.save_log:
                trajectory_images = [[] for _ in range(args.num_envs)]
            
            print("Rollout...")
            for step in tqdm(range(0, args.num_steps), desc="Rollout Progress"):
                global_step += args.num_envs
                
                # 供envs.step使用
                actions = np.zeros((args.num_envs, 7), dtype=np.float32)
                
                # 为每个环境单独调用agent（rollout函数目前只支持batch_size=1）
                # TODO: 向量化 Rollout
                for env_idx in range(args.num_envs):
                    prompt = infos['prompt'][env_idx]
                    
                    # 将numpy数组转换为PIL Image
                    obs_img = Image.fromarray(obs_imgs[env_idx].astype(np.uint8))
                    if obs_img.mode != 'RGB':
                        obs_img = obs_img.convert('RGB')
                    
                    # 收集图像到轨迹记录（仅在开启log时）
                    if args.save_log and trajectory_images is not None:
                        trajectory_images[env_idx].append(obs_img.copy())

                    # 处理单个环境的输入
                    query_inputs = agent.process_to_inputs(prompt, obs_img, args.max_seq_len)
                    
                    # 单独调用agent获取action
                    with torch.no_grad():
                        action, value, logprob, sequence, attention_mask = agent.act_rollout(query_inputs)
                        actions[env_idx] = action.squeeze(0)
                        # 直接写入到全局轨迹数据中
                        values[step, env_idx] = value.squeeze(0)
                        logprobs[step, env_idx] = logprob.squeeze(0)
                        assert sequence.shape[1] <= args.max_seq_len and attention_mask.shape[1] <= args.max_seq_len, \
                            f"sequence.shape[1]={sequence.shape[1]} > args.max_seq_len={args.max_seq_len} or attention_mask.shape[1]={attention_mask.shape[1]} > args.max_seq_len={args.max_seq_len}"
                        traj_sequences[step, env_idx, :sequence.shape[1]] = sequence.squeeze(0)
                        traj_attention_mask[step, env_idx, :attention_mask.shape[1]] = attention_mask.squeeze(0)
                        traj_obs[step, env_idx] = query_inputs['pixel_values'].squeeze(0)
                
                dones[step, :] = next_done

                obs_imgs, rewards_step, terminated_step, truncated_step, infos = envs.step(actions)
                # TODO: 截断的时候也视为 done 吗？如果被截断，next_value 不应该为 0
                next_done = torch.tensor(terminated_step | truncated_step, dtype=torch.int).to(device)
                # 记录奖励和成功状态
                rewards[step, :] = torch.tensor(rewards_step).to(device).view(-1)
                success[step, :] = infos['success']


            # 广义优势估计（GAE）
            with torch.no_grad():
                lastgaelam = torch.zeros(args.num_envs).to(device)
                
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        # TODO: bootstrap value if not terminated
                        nextvalues = torch.zeros(args.num_envs).to(device)
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            
            # 保存每个环境的日志数据
            if trajectory_logger is not None:
                trajectory_logger.save_trajectories(
                    iteration=iteration,
                    rewards=rewards,
                    success=success,
                    values=values,
                    logprobs=logprobs,
                    advantages=advantages,
                    returns=returns,
                    dones=dones,
                    trajectory_images=trajectory_images if trajectory_images is not None else []
                )
        
        # broadcast数据到所有进程
        dist.barrier()
        dist.broadcast(traj_obs, src=0)
        dist.broadcast(traj_sequences, src=0)
        dist.broadcast(traj_attention_mask, src=0)
        dist.broadcast(logprobs, src=0)
        dist.broadcast(rewards, src=0)
        dist.broadcast(dones, src=0)
        dist.broadcast(advantages, src=0)
        dist.broadcast(values, src=0)
        dist.broadcast(returns, src=0)


        # ==================================================================================================================
        # Stage 3: Update
        # ==================================================================================================================
        # NOTE: all the GPUs will do the update together
        # flatten the batch
        b_sequences = traj_sequences.reshape((-1, args.max_seq_len))
        b_attention_mask = traj_attention_mask.reshape((-1, args.max_seq_len))
        b_pixel_values = traj_obs.reshape((-1,) + observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 使用DataLoader和DistributedSampler进行数据采样
        # 创建PPO数据集
        dataset = PPODataset(
            sequences=b_sequences,
            attention_mask=b_attention_mask,
            pixel_values=b_pixel_values,
            logprobs=b_logprobs,
            advantages=b_advantages,
            returns=b_returns,
            values=b_values
        )
        
        # 创建DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=False
        )
        
        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=args.minibatch_size,
            sampler=sampler,
            num_workers=0,  # 设为0避免多进程问题
            pin_memory=False,  # 数据在GPU上，不需要pin_memory
            drop_last=False
        )

        # Optimizing the policy and value network
        clipfracs = []
        
        # 初始化变量以避免可能未定义的错误
        approx_kl = torch.tensor(0.0, device=device)
        old_approx_kl = torch.tensor(0.0, device=device)
        v_loss = torch.tensor(0.0, device=device)
        pg_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)
        
        print("Update...")
        for epoch in range(args.update_epochs):
            # 设置sampler的epoch，确保每个epoch的数据shuffle不同
            sampler.set_epoch(epoch)
            batch_count = 0
            rank_data_ids = []
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.update_epochs}"):
                # 从batch中提取数据
                mb_sequences = batch['sequences']
                mb_attention_mask = batch['attention_mask']
                mb_pixel_values = batch['pixel_values']
                mb_logprobs = batch['logprobs']
                mb_advantages = batch['advantages']
                mb_returns = batch['returns']
                mb_values = batch['values']
                mb_sample_ids = batch['sample_id']
                rank_data_ids.extend(mb_sample_ids.cpu().tolist())

                # 重新编码轨迹数据以计算新的log概率和values
                newlogprob, newvalue = agent.encode_traj(
                    mb_sequences, 
                    mb_attention_mask, 
                    mb_pixel_values, 
                )
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # 对优势值进行归一化
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                # TODO: entropy loss 暂时去掉，设置为0
                entropy_loss = torch.tensor(0.0, device=device)
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                batch_count += 1
            
            rank_data_ids.sort()
            print(f"Mini-batch finish: Rank={dist.get_rank()}, Sample IDs={rank_data_ids}")

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if envs is not None:
        envs.close()
    writer.close()


if __name__ == "__main__":
    main()
