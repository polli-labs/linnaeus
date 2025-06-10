import argparse
import os
import random  # For seed setting
import sys

import numpy as np
import torch
import torch.distributed as dist

# Linnaeus imports
try:
    from linnaeus.config import get_default_config, load_config, setup_output_dirs, update_config
    from linnaeus.h5data.build import build_datasets, build_loaders
    from linnaeus.models.base_model import BaseModel
    from linnaeus.models.build import build_model
    from linnaeus.rl_env.environment import TaxonomicClassificationEnv
    from linnaeus.rl_env.policies import LinnaeusPolicyWrapper
    from linnaeus.rl_env.reward_functions import AbstentionRewardFunction, EpisodeOutcomeReward, SimpleAbstentionReward
    from linnaeus.rl_env.verifier import TaxonomicRLVerifier  # Needed for explicit instantiation
    from linnaeus.utils.checkpoint import load_checkpoint
    from linnaeus.utils.config_utils import load_model_base_config
    from linnaeus.utils.distributed import get_rank_safely, get_world_size, init_distributed_mode, is_main_process_safely
    from linnaeus.utils.logging.logger import create_h5data_logger, create_logger, get_h5data_logger, get_main_logger
    from linnaeus.utils.logging.wandb import finish_wandb, initialize_wandb, log_to_wandb
    from linnaeus.utils.taxonomy.taxonomy_tree import TaxonomyTree
except ImportError as e:
    print(f"Failed to import Linnaeus components: {e}. Ensure PYTHONPATH is set correctly.")
    sys.exit(1)

# For PPO
from typing import Any

import torch.optim as optim
from torch import nn


# Helper functions for PPO (compute_gae_and_returns, ppo_update - assumed to be present from previous step)
def compute_gae_and_returns(
    rewards_batch: np.ndarray, values_batch: np.ndarray, dones_batch: np.ndarray,
    last_value_np: float, gamma: float, gae_lambda: float
) -> tuple[np.ndarray, np.ndarray]:
    num_steps = len(rewards_batch)
    advantages = np.zeros_like(rewards_batch)
    last_gae_lam = 0.0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = 1.0 - dones_batch[t]
            next_values = last_value_np
        else:
            next_non_terminal = 1.0 - dones_batch[t]
            next_values = values_batch[t+1]
        delta = rewards_batch[t] + gamma * next_values * next_non_terminal - values_batch[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
    returns = advantages + values_batch
    return advantages, returns

def ppo_update(
    policy_wrapper: LinnaeusPolicyWrapper, optimizer: optim.Optimizer, trajectory_data: list[dict[str, Any]],
    last_value_final_next_obs_np: float, gamma: float, gae_lambda: float, ppo_epochs: int, ppo_batch_size: int,
    clip_epsilon: float, vf_coef: float, ent_coef: float, max_grad_norm: float, logger: Any, device: torch.device,
    rl_mode: str, rank_order_for_multitask: list[str] | None = None
):
    obs_list = [t['obs'] for t in trajectory_data]
    actions_list = [t['action'] for t in trajectory_data]
    log_probs_old_np = np.array([t['log_prob'] for t in trajectory_data], dtype=np.float32)
    rewards_np = np.array([t['reward'] for t in trajectory_data], dtype=np.float32)
    dones_np = np.array([t['done'] for t in trajectory_data], dtype=np.float32)
    values_np = np.array([t['value'] for t in trajectory_data], dtype=np.float32)
    advantages_np, returns_np = compute_gae_and_returns(rewards_np, values_np, dones_np, last_value_final_next_obs_np, gamma, gae_lambda)
    advantages_np = (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)
    num_trajectory_steps = len(trajectory_data)
    indices = np.arange(num_trajectory_steps)
    policy_wrapper.train()
    for _ in range(ppo_epochs):
        np.random.shuffle(indices)
        for start_idx in range(0, num_trajectory_steps, ppo_batch_size):
            minibatch_indices = indices[start_idx : start_idx + ppo_batch_size]
            batch_images_list = [obs_list[i]['image'] for i in minibatch_indices]
            processed_images_list = []
            for img_np in batch_images_list:
                if img_np.ndim == 3 and img_np.shape[-1] == 3:
                    processed_images_list.append(np.transpose(img_np, (2,0,1)))
                elif img_np.ndim == 3 and img_np.shape[0] == 3:
                    processed_images_list.append(img_np)
                else:
                    raise ValueError(f"Unexpected image shape in minibatch: {img_np.shape}")
            mb_obs_images = torch.as_tensor(np.stack(processed_images_list), dtype=torch.float32, device=device)
            minibatch_obs_dict = {"image": mb_obs_images}
            mb_actions: torch.Tensor | list[torch.Tensor]
            if rl_mode == "sequential":
                batch_rank_indices_list = [obs_list[i]['current_rank_index'] for i in minibatch_indices]
                minibatch_obs_dict["current_rank_index"] = torch.as_tensor(batch_rank_indices_list, dtype=torch.long, device=device)
                mb_actions = torch.as_tensor(np.array([actions_list[i] for i in minibatch_indices]), dtype=torch.long, device=device)
            elif rl_mode == "multitask":
                if rank_order_for_multitask is None:
                    raise ValueError("rank_order_for_multitask cannot be None for multitask PPO update.")
                num_ranks = len(rank_order_for_multitask)
                actions_per_rank_for_batch = [[] for _ in range(num_ranks)]
                for i in minibatch_indices:
                    action_sample = actions_list[i]
                    for rank_j in range(num_ranks):
                        actions_per_rank_for_batch[rank_j].append(action_sample[rank_j])
                mb_actions = [torch.as_tensor(np.array(actions_for_one_rank), dtype=torch.long, device=device) for actions_for_one_rank in actions_per_rank_for_batch]
            else:
                raise ValueError(f"Unsupported rl_mode in ppo_update: {rl_mode}")
            mb_log_probs_old = torch.as_tensor(log_probs_old_np[minibatch_indices], dtype=torch.float32, device=device)
            mb_advantages = torch.as_tensor(advantages_np[minibatch_indices], dtype=torch.float32, device=device)
            mb_returns = torch.as_tensor(returns_np[minibatch_indices], dtype=torch.float32, device=device)
            new_values, new_log_probs, entropy = policy_wrapper.evaluate_actions(minibatch_obs_dict, mb_actions)
            ratio = torch.exp(new_log_probs - mb_log_probs_old)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(new_values, mb_returns)
            entropy_loss = -entropy.mean()
            total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
            optimizer.zero_grad()
            total_loss.backward()
            if max_grad_norm > 0:  # Apply gradient clipping if max_grad_norm is positive
                torch.nn.utils.clip_grad_norm_(policy_wrapper.parameters(), max_norm=max_grad_norm)
            optimizer.step()
    policy_wrapper.eval()
    if is_main_process_safely():
        log_data_ppo = {
            "ppo/policy_loss": policy_loss.item(),
            "ppo/value_loss": value_loss.item(),
            "ppo/entropy": -entropy_loss.item(),
            "ppo/total_loss": total_loss.item()
        }
        logger.info(f"PPO Update: Losses - Policy={log_data_ppo['ppo/policy_loss']:.4f}, Value={log_data_ppo['ppo/value_loss']:.4f}, Entropy={log_data_ppo['ppo/entropy']:.4f}")
        if initialize_wandb.wandb_is_initialized: log_to_wandb(log_data_ppo)

def evaluate_policy(
    policy_wrapper: LinnaeusPolicyWrapper, eval_env: TaxonomicClassificationEnv, num_eval_episodes: int,
    logger: Any, device: torch.device, rl_mode: str, rank_order_for_multitask: list[str] | None = None,
):
    policy_wrapper.eval()
    episode_rewards, episode_lengths = [], []
    abstention_counts_per_rank = {r: {"abstained":0,"total_decisions":0,"correct_abstain":0,"unnecessary_abstain":0,
                                      "missed_abstain":0,"correct_classify_non_abstain":0,"misclassify_non_abstain":0}
                                  for r in eval_env.rank_order}
    for i_episode in range(num_eval_episodes):
        obs_dict, info_dict = eval_env.reset()
        done, truncated = False, False
        current_eval_episode_reward, current_eval_episode_length = 0.0, 0
        gt_full_sequence = info_dict.get("ground_truth", {})
        while not (done or truncated):
            img_np = obs_dict['image']
            if img_np.ndim == 3 and img_np.shape[-1] == 3:
                img_np = np.transpose(img_np, (2,0,1))
            image_tensor = torch.as_tensor(img_np, dtype=torch.float32, device=device).unsqueeze(0)
            policy_input_obs = {"image": image_tensor}
            if rl_mode == "sequential":
                policy_input_obs["current_rank_index"] = torch.as_tensor([obs_dict['current_rank_index']], dtype=torch.long, device=device)
            with torch.no_grad():
                action_dist, _ = policy_wrapper(policy_input_obs)
            action_to_env: int | np.ndarray
            if rl_mode == "sequential":
                action_to_env = torch.argmax(action_dist.logits, dim=-1).item()
            elif rl_mode == "multitask":
                action_to_env = torch.stack([torch.argmax(d.logits, dim=-1) for d in action_dist], dim=0).T.squeeze(0).cpu().numpy()
            obs_dict, reward, done, truncated, step_info = eval_env.step(action_to_env)
            current_eval_episode_reward += reward
            current_eval_episode_length += 1
            if (rl_mode == "sequential" and done) or (rl_mode == "multitask"):
                if 'final_predictions' in step_info:
                    for r_name, p_list in step_info['final_predictions'].items():
                        if r_name not in abstention_counts_per_rank:
                            continue
                        counts = abstention_counts_per_rank[r_name]
                        counts["total_decisions"] += 1
                        gt_label = gt_full_sequence.get(r_name, [None])[0]
                        pred_label = p_list[0]
                        if pred_label is None:
                            counts["abstained"] += 1
                            if gt_label is None:
                                counts["correct_abstain"] += 1
                            else:
                                counts["unnecessary_abstain"] += 1
                        else:
                            if gt_label is None:
                                counts["missed_abstain"] += 1
                            elif pred_label == gt_label:
                                counts["correct_classify_non_abstain"] += 1
                            else:
                                counts["misclassify_non_abstain"] +=1
                if rl_mode == "sequential":
                    break
        episode_rewards.append(current_eval_episode_reward)
        episode_lengths.append(current_eval_episode_length)
        if is_main_process_safely():
            logger.debug(f"Eval Ep {i_episode+1}/{num_eval_episodes}: R={current_eval_episode_reward:.2f}, L={current_eval_episode_length}")
    policy_wrapper.train()
    eval_results = {"mean_reward": np.mean(episode_rewards) if episode_rewards else 0,
                    "median_reward": np.median(episode_rewards) if episode_rewards else 0,
                    "mean_length": np.mean(episode_lengths) if episode_lengths else 0}
    for r_name, c in abstention_counts_per_rank.items():
        total = c["total_decisions"]
        eval_results[f"abst_rate/{r_name}"] = (c["abstained"]/total) if total > 0 else 0
        eval_results[f"corr_abst_rate/{r_name}"]=(c["correct_abstain"]/c["abstained"]) if c["abstained"]>0 else 0
        eval_results[f"unnec_abst_rate/{r_name}"]=(c["unnecessary_abstain"]/c["abstained"]) if c["abstained"]>0 else 0
        should_abst = c["correct_abstain"]+c["missed_abstain"]
        eval_results[f"recall_abst/{r_name}"]=(c["correct_abstain"]/should_abst) if should_abst > 0 else 0
        did_classify_non_null = c["correct_classify_non_abstain"]+c["misclassify_non_abstain"]
        eval_results[f"acc_on_non_abst/{r_name}"]=(c["correct_classify_non_abstain"]/did_classify_non_null) if did_classify_non_null > 0 else 0
        eval_results[f"missed_abst_count/{r_name}"]=c["missed_abstain"]
    return eval_results

def parse_args(args_list=None):
    parser = argparse.ArgumentParser("Linnaeus RL Abstention Training", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="Path to main config file for RL training")
    parser.add_argument("--phase1_model_path", type=str, required=True, help="Path to the pretrained Phase 1 model checkpoint")
    parser.add_argument("--phase1_model_cfg", type=str, help="Path to the Phase 1 model's original config file")
    parser.add_argument("--opts", help="Command line MargeList options to override config", default=None, nargs=argparse.REMAINDER)
    args, _ = parser.parse_known_args(args_list)
    config = get_default_config()
    if os.path.exists(args.cfg):
        exp_config = load_config(args.cfg)
        config.merge_from_other_cfg(exp_config)
    else:
        raise FileNotFoundError(f"RL Training config file not found: {args.cfg}")
    if args.opts:
        config.merge_from_list(args.opts)
    config.defrost()
    if not config.ENV.OUTPUT.ROOT:
        config.ENV.OUTPUT.ROOT = os.path.join(os.getcwd(), "output", "rl_training")
    if "_RL_Abstention" not in config.ENV.EXPERIMENT_NAME:
        config.ENV.EXPERIMENT_NAME = config.ENV.EXPERIMENT_NAME + "_RL_Abstention" if config.ENV.EXPERIMENT_NAME else "RL_Abstention_Training"

    # Ensure TRAIN.RL node and its sub-nodes/defaults exist
    if not hasattr(config.TRAIN, 'RL'):
        config.TRAIN.RL = {}
    config.TRAIN.RL.defrost()
    config.TRAIN.RL.MODE = config.TRAIN.RL.get("MODE", "sequential")
    config.TRAIN.RL.TOTAL_TIMESTEPS = config.TRAIN.RL.get("TOTAL_TIMESTEPS", 1_000_000)
    config.TRAIN.RL.STEPS_PER_BATCH = config.TRAIN.RL.get("STEPS_PER_BATCH", 2048)
    config.TRAIN.RL.POLICY_DEVICE = config.TRAIN.RL.get("POLICY_DEVICE", "cuda")
    config.TRAIN.RL.LOG_INTERVAL_BATCHES = config.TRAIN.RL.get("LOG_INTERVAL_BATCHES", 1)  # Log after every PPO update batch
    config.TRAIN.RL.EVAL_INTERVAL_BATCHES = config.TRAIN.RL.get("EVAL_INTERVAL_BATCHES", 10)
    config.TRAIN.RL.NUM_EVAL_EPISODES = config.TRAIN.RL.get("NUM_EVAL_EPISODES", 10)
    config.TRAIN.RL.LEARNING_RATE = config.TRAIN.RL.get("LEARNING_RATE", 3e-4)
    config.TRAIN.RL.FINETUNE_STRATEGY = config.TRAIN.RL.get("FINETUNE_STRATEGY", "heads_only")
    config.TRAIN.RL.NUM_UNFROZEN_BACKBONE_BLOCKS = config.TRAIN.RL.get("NUM_UNFROZEN_BACKBONE_BLOCKS", 2)
    if not hasattr(config.TRAIN.RL, 'PPO'):
        config.TRAIN.RL.PPO = {}
    config.TRAIN.RL.PPO.defrost()
    config.TRAIN.RL.PPO.EPOCHS = config.TRAIN.RL.PPO.get("EPOCHS", 4)
    config.TRAIN.RL.PPO.BATCH_SIZE = config.TRAIN.RL.PPO.get("BATCH_SIZE", 64)
    config.TRAIN.RL.PPO.GAMMA = config.TRAIN.RL.PPO.get("GAMMA", 0.99)
    config.TRAIN.RL.PPO.GAE_LAMBDA = config.TRAIN.RL.PPO.get("GAE_LAMBDA", 0.95)
    config.TRAIN.RL.PPO.CLIP_EPSILON = config.TRAIN.RL.PPO.get("CLIP_EPSILON", 0.2)
    config.TRAIN.RL.PPO.VF_COEF = config.TRAIN.RL.PPO.get("VF_COEF", 0.5)
    config.TRAIN.RL.PPO.ENT_COEF = config.TRAIN.RL.PPO.get("ENT_COEF", 0.01)
    config.TRAIN.RL.PPO.MAX_GRAD_NORM = config.TRAIN.RL.PPO.get("MAX_GRAD_NORM", 0.5)
    config.TRAIN.RL.PPO.freeze()
    if not hasattr(config.TRAIN.RL, 'REWARD_FUNCTION'):
        config.TRAIN.RL.REWARD_FUNCTION = {}
    config.TRAIN.RL.REWARD_FUNCTION.defrost()
    config.TRAIN.RL.REWARD_FUNCTION.TYPE = config.TRAIN.RL.REWARD_FUNCTION.get("TYPE", "SimpleAbstentionReward")
    if not hasattr(config.TRAIN.RL.REWARD_FUNCTION, 'PARAMS'):
        config.TRAIN.RL.REWARD_FUNCTION.PARAMS = {}
    sar_params = config.TRAIN.RL.REWARD_FUNCTION.PARAMS
    sar_params.defrost()
    sar_params["reward_correct_classification"]=sar_params.get("reward_correct_classification",1.0)
    sar_params["reward_correct_abstention"]=sar_params.get("reward_correct_abstention",0.5)
    sar_params["penalty_misclassification"]=sar_params.get("penalty_misclassification",-1.0)
    sar_params["penalty_unnecessary_abstention"]=sar_params.get("penalty_unnecessary_abstention",-0.5)
    sar_params["penalty_incorrect_prediction_at_null_rank"]=sar_params.get("penalty_incorrect_prediction_at_null_rank",-1.0)
    sar_params.freeze()
    config.TRAIN.RL.REWARD_FUNCTION.freeze()
    config.TRAIN.RL.freeze()
    if not hasattr(config.MODEL, 'RL_POLICY'):
        config.MODEL.RL_POLICY = {}
    config.MODEL.RL_POLICY.defrost()
    config.MODEL.RL_POLICY.BACKBONE_FEATURES_DIM = config.MODEL.RL_POLICY.get("BACKBONE_FEATURES_DIM", 512)
    config.MODEL.RL_POLICY.freeze()
    config.freeze()
    config = update_config(config, args)
    config = setup_output_dirs(config, args=args)
    return config, args

def main():
    config, cmd_args = parse_args() # cmd_args now simpler
    init_distributed_mode(config)
    rank = get_rank_safely(); world_size = get_world_size()
    create_logger(output_dir=config.ENV.OUTPUT.DIRS.LOGS, dist_rank=rank, name="", log_level=config.EXPERIMENT.LOG_LEVEL_MAIN)
    create_h5data_logger(output_dir=config.ENV.OUTPUT.DIRS.LOGS, dist_rank=rank, log_level=config.EXPERIMENT.LOG_LEVEL_H5DATA)
    logger = get_main_logger()
    logger.info(f"RL Training. Rank: {rank}, World: {world_size}, Output: {config.ENV.OUTPUT.DIRS.ROOT}")
    if rank == 0 and config.EXPERIMENT.WANDB.ENABLED:
        logger.info("Initializing WandB...")
        initialize_wandb(config=config, model=None)  # Pass policy_wrapper later

    seed = config.MISC.SEED + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("Building data components...")
    h5_logger = get_h5data_logger()
    dataset_train,_,num_classes_supervised,_,_,taxonomy_tree,_,_,_,_,_ = build_datasets(config,h5_logger,monitor_enabled=False)
    if dataset_train is None or taxonomy_tree is None:
        logger.error("Dataset/TaxonomyTree build failed.")
        sys.exit(1)
    train_loader,_=build_loaders(config,dataset_train,None,h5_logger)
    logger.info("Data components built.")

    logger.info(f"Loading Phase 1 model from: {cmd_args.phase1_model_path}")
    phase1_cfg_path = cmd_args.phase1_model_cfg or config.MODEL.RL_POLICY.get("PHASE1_MODEL_CFG")  # Allow in RL config too
    phase1_cfg_for_build = get_default_config()
    if phase1_cfg_path and os.path.exists(phase1_cfg_path):
        phase1_cfg_for_build.merge_from_other_cfg(load_config(phase1_cfg_path))
    else:
        phase1_cfg_for_build.merge_from_other_cfg(config)  # Fallback to current config structure
    phase1_cfg_for_build.defrost()
    phase1_cfg_for_build.MODEL.PRETRAINED=None
    phase1_cfg_for_build.MODEL.RESUME=None
    phase1_cfg_for_build.freeze()
    phase1_model:BaseModel=build_model(config=phase1_cfg_for_build,num_classes=num_classes_supervised,taxonomy_tree=taxonomy_tree)
    load_checkpoint(phase1_cfg_for_build,phase1_model,logger,ckpt_path=cmd_args.phase1_model_path,strict=False)
    policy_device=torch.device(config.TRAIN.RL.POLICY_DEVICE if torch.cuda.is_available() and config.TRAIN.RL.POLICY_DEVICE=="cuda" else "cpu")
    phase1_model.to(policy_device)
    phase1_model.eval()
    logger.info(f"Phase 1 model loaded to {policy_device}")

    reward_fn_type = config.TRAIN.RL.REWARD_FUNCTION.TYPE
    reward_fn_params = config.TRAIN.RL.REWARD_FUNCTION.PARAMS
    reward_function_instance: AbstentionRewardFunction | None = None
    if reward_fn_type == "SimpleAbstentionReward":
        reward_function_instance = SimpleAbstentionReward(**reward_fn_params)
    elif reward_fn_type == "EpisodeOutcomeReward":
        reward_function_instance = EpisodeOutcomeReward(**reward_fn_params)
    else:
        raise ValueError(f"Unsupported reward function: {reward_fn_type}")
    logger.info(f"Using reward function: {reward_fn_type} with params: {reward_fn_params}")

    verifier_instance = TaxonomicRLVerifier(taxonomy_tree=taxonomy_tree, reward_function=reward_function_instance, rank_order=taxonomy_tree.task_keys)
    rl_env_mode = config.TRAIN.RL.MODE
    rl_env = TaxonomicClassificationEnv(
        dataloader=train_loader, taxonomy_tree=taxonomy_tree, verifier=verifier_instance, mode=rl_env_mode,
        image_shape=tuple(config.DATA.get("IMAGE_SHAPE", (3,224,224)))
    )
    logger.info(f"RL Env '{rl_env_mode}' mode initialized.")

    backbone_feat_dim = config.MODEL.RL_POLICY.BACKBONE_FEATURES_DIM
    if backbone_feat_dim is None:
        if hasattr(phase1_model,'backbone_feat_dim'):
            backbone_feat_dim = phase1_model.backbone_feat_dim
        else:
            logger.error("BACKBONE_FEATURES_DIM not set/inferable.")
            sys.exit(1)
    policy_num_classes = {name: count+1 for name,count in taxonomy_tree.num_classes.items()}
    policy_wrapper = LinnaeusPolicyWrapper(
        linnaeus_model=phase1_model, backbone_features_dim=backbone_feat_dim,
        num_classes_at_rank=policy_num_classes, mode=rl_env_mode, rank_order=taxonomy_tree.task_keys
    )
    policy_wrapper.to(policy_device)
    logger.info(f"Policy wrapper on {policy_device}")
    if rank == 0 and config.EXPERIMENT.WANDB.ENABLED and not getattr(initialize_wandb, 'wandb_is_initialized', False):
        initialize_wandb(config=config, model=policy_wrapper)

    finetune_strat = config.TRAIN.RL.FINETUNE_STRATEGY
    logger.info(f"Fine-tuning strategy: {finetune_strat}")
    if finetune_strat == "value_head_only":
        for param in policy_wrapper.linnaeus_model.parameters():
            param.requires_grad = False
        logger.info("Frozen Linnaeus model. Training value_head only.")
    elif finetune_strat == "heads_only":
        if hasattr(policy_wrapper.linnaeus_model, 'backbone') and policy_wrapper.linnaeus_model.backbone:
            for param in policy_wrapper.linnaeus_model.backbone.parameters():
                param.requires_grad = False
            logger.info("Frozen Linnaeus backbone.")
        else:
            logger.warning("No 'backbone' on Linnaeus model for 'heads_only'.")
        for attr in ['heads','head','classification_heads']:
            if hasattr(policy_wrapper.linnaeus_model,attr):
                module=getattr(policy_wrapper.linnaeus_model,attr)
                if isinstance(module,nn.Module):
                    [p.requires_grad_(True) for p in module.parameters()]
                    logger.info(f"Ensured '{attr}' trainable.")
                    break
    elif finetune_strat == "last_n_blocks":
        num_unfrozen = config.TRAIN.RL.NUM_UNFROZEN_BACKBONE_BLOCKS
        if hasattr(policy_wrapper.linnaeus_model,'backbone') and hasattr(policy_wrapper.linnaeus_model.backbone,'blocks') \
           and isinstance(policy_wrapper.linnaeus_model.backbone.blocks,nn.ModuleList) and len(policy_wrapper.linnaeus_model.backbone.blocks)>0:
            total_blocks = len(policy_wrapper.linnaeus_model.backbone.blocks)
            for param in policy_wrapper.linnaeus_model.backbone.parameters():
                param.requires_grad = False
            if 0 < num_unfrozen <= total_blocks:
                for i in range(total_blocks-num_unfrozen,total_blocks):
                    [p.requires_grad_(True) for p in policy_wrapper.linnaeus_model.backbone.blocks[i].parameters()]
                logger.info(f"Froze backbone, unfroze last {num_unfrozen} blocks.")
            elif num_unfrozen > total_blocks:
                [p.requires_grad_(True) for p in policy_wrapper.linnaeus_model.backbone.parameters()]
                logger.warning("Unfreezing all backbone blocks.")
            else:
                logger.info("All backbone blocks frozen.")
        else:
            logger.warning("'last_n_blocks' failed. Defaulting to 'heads_only'.")
            if hasattr(policy_wrapper.linnaeus_model,'backbone') and policy_wrapper.linnaeus_model.backbone:
                [p.requires_grad_(False) for p in policy_wrapper.linnaeus_model.backbone.parameters()]
    elif finetune_strat == "full":
        [p.requires_grad_(True) for p in policy_wrapper.linnaeus_model.parameters()]
        logger.info("Full fine-tuning enabled.")
    else:
        logger.warning(f"Unknown strategy: {finetune_strat}. Defaulting to 'heads_only'.")
        if hasattr(policy_wrapper.linnaeus_model,'backbone') and policy_wrapper.linnaeus_model.backbone:
            [p.requires_grad_(False) for p in policy_wrapper.linnaeus_model.backbone.parameters()]

    trainable_params = filter(lambda p: p.requires_grad, policy_wrapper.parameters())
    optimizer = optim.Adam(trainable_params, lr=config.TRAIN.RL.LEARNING_RATE)
    logger.info(f"Optimizer with LR {config.TRAIN.RL.LEARNING_RATE} for trainable params.")
    if is_main_process_safely():
        total_p = sum(p.numel() for p in policy_wrapper.parameters())
        train_p = sum(p.numel() for p in policy_wrapper.parameters() if p.requires_grad)
        logger.info(f"Policy Wrapper Params: Total={total_p}, Trainable={train_p} ({train_p/total_p*100:.2f}%)")

    logger.info("Starting PPO training loop...")
    traj_batch, ep_rewards, ep_lengths = [], [], []
    obs_dict, _ = rl_env.reset()
    current_ep_reward, current_ep_length, ep_this_batch = 0.0,0,0
    ppo_updates_done = 0

    for overall_ts in range(1, config.TRAIN.RL.TOTAL_TIMESTEPS + 1):
        img_np = obs_dict['image']
        if img_np.ndim == 3 and img_np.shape[-1] == 3:
            img_np = np.transpose(img_np, (2,0,1))
        img_tensor = torch.as_tensor(img_np,dtype=torch.float32,device=policy_device).unsqueeze(0)
        pol_input_obs = {"image":img_tensor}
        if rl_env_mode == "sequential":
            pol_input_obs["current_rank_index"]=torch.as_tensor([obs_dict['current_rank_index']],dtype=torch.long,device=policy_device)
        with torch.no_grad():
            action_dist, val_tensor = policy_wrapper(pol_input_obs)
        action_env:int | np.ndarray
        if rl_env_mode == "sequential":
            action_t=action_dist.sample()
            log_prob_t=action_dist.log_prob(action_t)
            action_env=action_t.item()
        elif rl_env_mode=="multitask":
            if not isinstance(action_dist,list):
                raise TypeError(f"Expected list of dists, got {type(action_dist)}")
            s_actions=[d.sample() for d in action_dist]
            action_t=torch.stack(s_actions,dim=0).T
            log_probs_r=[d.log_prob(a) for d,a in zip(action_dist,s_actions, strict=False)]
            log_prob_t=torch.stack(log_probs_r,dim=1).sum(dim=1)
            action_env=action_t.squeeze(0).cpu().numpy()
        else:
            raise ValueError(f"Unsupported RL mode: {rl_env_mode}")
        next_obs_dict,rew,done,trunc,info_step=rl_env.step(action_env)
        traj_batch.append({"obs":obs_dict,"action":action_env,"reward":rew,"done":float(done),
                           "log_prob":log_prob_t.detach().cpu().numpy().item(),
                           "value":val_tensor.detach().cpu().numpy().item(),"next_obs":next_obs_dict})
        obs_dict=next_obs_dict
        current_ep_reward+=rew
        current_ep_length+=1
        if done or trunc:
            reason="trunc" if trunc and not done else info_step.get('reason_for_done','end')
            if is_main_process_safely():
                logger.debug(f"T{overall_ts}: Ep end R={current_ep_reward:.2f} L={current_ep_length} {reason}")
            ep_rewards.append(current_ep_reward)
            ep_lengths.append(current_ep_length)
            obs_dict,_=rl_env.reset()
            current_ep_reward=0.0
            current_ep_length=0
            ep_this_batch+=1
        if len(traj_batch) >= config.TRAIN.RL.STEPS_PER_BATCH:
            last_val=0.0
            if not (done or trunc):
                img_f_np=obs_dict['image']
                if img_f_np.ndim==3 and img_f_np.shape[-1]==3:
                    img_f_np=np.transpose(img_f_np,(2,0,1))
                img_f_tensor=torch.as_tensor(img_f_np,dtype=torch.float32,device=policy_device).unsqueeze(0)
                obs_f_pol={"image":img_f_tensor}
                if rl_env_mode=="sequential":
                    obs_f_pol["current_rank_index"]=torch.as_tensor([obs_dict['current_rank_index']],dtype=torch.long,device=policy_device)
                with torch.no_grad():
                    _,last_val_t=policy_wrapper(obs_f_pol)
                    last_val=last_val_t.detach().cpu().numpy().item()
            if is_main_process_safely():
                logger.info(f"Collected {len(traj_batch)} steps. Last GAE val: {last_val:.3f}. Updating PPO...")
            ppo_update(policy_wrapper,optimizer,traj_batch,last_val,
                       config.TRAIN.RL.PPO.GAMMA,config.TRAIN.RL.PPO.GAE_LAMBDA,config.TRAIN.RL.PPO.EPOCHS,config.TRAIN.RL.PPO.BATCH_SIZE,
                       config.TRAIN.RL.PPO.CLIP_EPSILON,config.TRAIN.RL.PPO.VF_COEF,config.TRAIN.RL.PPO.ENT_COEF,config.TRAIN.RL.PPO.MAX_GRAD_NORM,
                       logger,policy_device,rl_env_mode,taxonomy_tree.task_keys if rl_env_mode=="multitask" else None)
            traj_batch=[]
            ep_this_batch=0
            ppo_updates_done+=1
            if rank==0 and config.EXPERIMENT.WANDB.ENABLED and getattr(initialize_wandb,'wandb_is_initialized',False):
                log_data={"rl_train/mean_episode_reward":np.mean(ep_rewards) if ep_rewards else 0,
                          "rl_train/median_episode_reward":np.median(ep_rewards) if ep_rewards else 0,
                          "rl_train/mean_episode_length":np.mean(ep_lengths) if ep_lengths else 0,
                          "rl_train/ppo_update_count":ppo_updates_done,
                          "rl_train/learning_rate":optimizer.param_groups[0]['lr']}
                log_to_wandb(log_data,step=overall_ts)
            if is_main_process_safely() and ep_rewards:
                logger.info(f"PPO Update #{ppo_updates_done}. Avg R: {np.mean(ep_rewards):.2f}, Avg L: {np.mean(ep_lengths):.2f}")
            ep_rewards,ep_lengths=[],[]
            if ppo_updates_done>0 and ppo_updates_done % config.TRAIN.RL.EVAL_INTERVAL_BATCHES == 0:
                if is_main_process_safely():
                    logger.info(f"Eval at PPO update #{ppo_updates_done} (Timestep {overall_ts})")
                eval_metrics=evaluate_policy(policy_wrapper,rl_env,config.TRAIN.RL.NUM_EVAL_EPISODES,logger,policy_device,
                                             rl_env_mode,taxonomy_tree.task_keys if rl_env_mode=="multitask" else None)
                if is_main_process_safely():
                    logger.info(f"Evaluation results: {eval_metrics}")
                    if config.EXPERIMENT.WANDB.ENABLED and getattr(initialize_wandb,'wandb_is_initialized',False):
                        log_to_wandb({f"rl_eval/{k}":v for k,v in eval_metrics.items()},step=overall_ts)
    logger.info("RL training loop finished.")
    if rank==0 and config.EXPERIMENT.WANDB.ENABLED and getattr(initialize_wandb,'wandb_is_initialized',False):
        finish_wandb()
    rl_env.close()
    logger.info("RL Env closed.")

if __name__ == "__main__":
    main_logger = None
    try:
        main()
    except Exception as e:
        main_logger = get_main_logger()
        if main_logger and main_logger.handlers:
            main_logger.error("Unhandled exception:", exc_info=True)
        else:
            print(f"FATAL (logger N/A): {e}")
            import traceback
            traceback.print_exc()
        if not isinstance(e, SystemExit) or (isinstance(e, SystemExit) and e.code == 0):
            sys.exit(1)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
