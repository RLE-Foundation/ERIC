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
import matplotlib.pyplot as plt
import mediapy as media
import torch.distributed as dist
import tensorflow as tf
import gym
import time

from accelerate import PartialState
from dataclasses import dataclass
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

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
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # VLA arguments
    vla_model_path: str = "openvla_models/openvla-7b-finetuned-libero-spatial"
    """the path to the VLA model"""
    lora_rank: int = 32
    """the rank of the LoRA matrix"""
    lora_dropout: float = 0.0
    """the dropout rate of the LoRA matrix"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
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
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

class LiberoEnv(gym.Env):
    def __init__(self, task_suite_name, task_id, seed, max_steps, init_state_id):
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
        self.seed = seed
        self.max_steps = max_steps
        self.init_state_id = init_state_id
        
        # Create the environment
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        task = task_suite.get_task(task_id)
        self.task_name = task.name
        self.task_description = task.language
        self.prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"
        self.prompt = self.prompt.replace("<INSTRUCTION>", self.task_description)
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self.env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=128, camera_widths=128)
        self.env.seed(seed)

        # Get the initial states
        self.initial_states = task_suite.get_task_init_states(task_id)
        self.init_state = self.initial_states[init_state_id]

        # Formulate observation space and action space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(6, 224, 224), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

        # Set a step counter
        self.current_step = 0
        
    def step(self, action):
        action = self.calibrate_action(action)
        obs, reward, done, info = self.env.step(action)
        visual_obs = self.get_visual_obs(obs)

        # Update the current step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        
        # Update the info
        info["prompt"] = self.prompt

        return visual_obs, reward, done, info
    
    def reset(self):
        # Reset the environment
        self.env.reset()
        # Set the initial state
        self.env.set_state(self.init_state)
        
        # Run 10 dummy steps to wait for the environment to stabilize
        dummy_action = [0.] * 7
        for _ in range(10):
            obs, reward, done, info = self.env.step(dummy_action)
        
        # Get the visual observation
        visual_obs = self.get_visual_obs(obs)

        # Update the info
        info['prompt'] = self.prompt

        return visual_obs

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
    
    def get_visual_obs(self, obs, resize_size=(224, 224)):
        """
        Get the visual observation from the environment.

        Args:
            obs (dict): Observation from the environment.
            resize_size (tuple): Resize size of the image.

        Returns:
            np.ndarray: Visual observation.
        """
        agentview_image = obs['agentview_image']
        # NOTE: rotate 180 degrees to match train preprocessing
        agentview_image = agentview_image[::-1, ::-1]
        # Resize to image size expected by model
        assert isinstance(resize_size, tuple)
        img = tf.image.encode_jpeg(agentview_image)
        img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
        img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
        img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
        img = Image.fromarray(img.numpy())
        img = img.convert("RGB")

        return img


def make_env(task_suite_name, task_id, seed, max_steps, init_state_id):
    def thunk():
        env = LiberoEnv(task_suite_name, task_id, seed, max_steps, init_state_id)
        return env
    return thunk

class Agent(nn.Module):
    def __init__(self, vla_base, action_tokenizer):
        super().__init__()

        self.vla_base = vla_base
        self.action_tokenizer = action_tokenizer

        self.config = vla_base.module.config.text_config
        self.config.n_embd = self.config.hidden_size if hasattr(
            self.config, "hidden_size") else self.config.n_embd

        self.value_head = nn.Sequential(
            nn.Linear(self.config.n_embd, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 1, bias=False),
        )
    
    def get_value(self, input_ids, attention_mask, pixel_values, **kwargs,):
        transformer_outputs = self.value_vla_base(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values,    # [B, 6, H, W]
            output_hidden_states=True,
            **kwargs,
        )

        hidden_states = transformer_outputs.hidden_states[-1] # [B, L, D], e.g. [2, 292, 4096]
        hidden_states = hidden_states[:, -1, :].float()  # [B, D], e.g. [2, 4096]

        values = self.value_head(hidden_states)

        return values
    
    def evaluate(self, input_ids, attention_mask, pixel_values, **kwargs):
        pass

    def act(self, query_inputs, **kwargs):
        input_ids = query_inputs['input_ids']
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
        query_inputs['input_ids'] = input_ids
        outputs = self.vla_base.module.generate(
            input_ids=query_inputs['input_ids'],
            attention_mask=query_inputs['attention_mask'],
            pixel_values=query_inputs['pixel_values'],
            max_new_tokens=7,
            do_sample=True,
            top_p=1.0,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_logits=True,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1][-1][:, 0, :]
        action_token_ids = outputs.sequences[:, -7:]
        logits = torch.concat(outputs.logits, dim=0)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs.gather(dim=-1, index=action_token_ids)
        actions = self.action_tokenizer.decode_token_ids_to_actions(action_token_ids.cpu().numpy())

        with torch.autocast("cuda", dtype=torch.bfloat16):
            values = self.value_head(hidden_states)

        return actions, action_token_ids, values, log_probs

if __name__ == "__main__":
    # DDP setup
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

    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    if distributed_state.is_main_process:
        envs = LiberoEnv(task_suite_name='libero_spatial', task_id=0, seed=0, max_steps=220, init_state_id=0)

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
    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # agent setup
    agent = Agent(vla_base, action_tokenizer).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    if distributed_state.is_main_process:
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    if distributed_state.is_main_process:
        next_obs = envs.reset()
        # next_obs = torch.Tensor(next_obs).to(device)
        next_query_inputs = processor(envs.prompt, next_obs).to("cuda:0", dtype=torch.bfloat16)
        next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # NOTE: only the main process will do the action
        if distributed_state.is_main_process:
            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step, :] = next_query_inputs['pixel_values']
                dones[step, :] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, action_token_ids, value, logprob = agent.act(next_query_inputs)
                    values[step, :] = value.flatten()
                actions[step, :] = action_token_ids
                logprobs[step, :] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, dones, infos = envs.step(action.cpu().numpy())
                rewards[step, :] = torch.tensor(reward).to(device).view(-1)
                next_query_inputs = processor(envs.prompt, next_obs).to("cuda:0", dtype=torch.bfloat16)
                next_done = torch.Tensor(dones).to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = torch.zeros(1, 1).to(device) # agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
        
        # NOTE: all the GPUs will do the update together
        dist.barrier()  
        torch.distributed.broadcast(obs, src=0)
        torch.distributed.broadcast(actions, src=0)
        torch.distributed.broadcast(logprobs, src=0)
        torch.distributed.broadcast(rewards, src=0)
        torch.distributed.broadcast(advantages, src=0)
        torch.distributed.broadcast(values, src=0)

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 这里的数据采样要使用dataloader+DistributedSampler

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, 2):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                newlogprob = torch.clone(b_logprobs[mb_inds])
                newvalue = torch.clone(b_values[mb_inds])
                entropy = torch.scalar_tensor(0.0)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

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

    envs.close()
    writer.close()
