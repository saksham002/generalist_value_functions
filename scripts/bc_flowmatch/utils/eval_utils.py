import torch
import numpy as np
from torchvision.transforms import Resize
import cv2
import imageio
from tqdm import tqdm
from einops import rearrange
from torch.utils.data import DataLoader
from gymnasium.vector import AsyncVectorEnv
import queue

def get_state(obs, horizon=1):
    state_history = []
    for t in range(horizon):
        state = np.concat([
            # obs['state']['left/ego_tcp_pose'][t], obs['state']['left/ego_tcp_vel'][t],
            obs["state"]["left/relative2_tcp_pose"][t], obs["state"]["left/relative2_tcp_vel"][t],
            # obs["state"]["left/wrist_tcp_pose"][t],
            obs["state"]["left/wrist_tcp_vel"][t],
            # obs["state"]["left/tcp_vel"][t],
            obs['state']['left/gripper_pos'][t],
            # obs['state']['right/ego_tcp_pose'][t], obs['state']['right/ego_tcp_vel'][t],
            obs["state"]["right/relative2_tcp_pose"][t], obs["state"]["right/relative2_tcp_vel"][t],
            # obs["state"]["right/wrist_tcp_pose"][t],
            obs["state"]["right/wrist_tcp_vel"][t],
            # obs["state"]["right/tcp_vel"][t],
            obs['state']['right/gripper_pos'][t],
        ])
        state_history.append(state)
    state_history = np.stack(state_history, axis=0)
    return torch.from_numpy(state_history).float().unsqueeze(0)

def get_state_batched(obs):
    state = np.concat([
        # obs['state']['left/ego_tcp_pose'][t], obs['state']['left/ego_tcp_vel'][t],
        obs["state"]["left/relative2_tcp_pose"], obs["state"]["left/relative2_tcp_vel"],
        # obs["state"]["left/wrist_tcp_pose"][t],
        obs["state"]["left/wrist_tcp_vel"],
        obs['state']['left/gripper_pos'],
        # obs['state']['right/ego_tcp_pose'][t], obs['state']['right/ego_tcp_vel'][t],
        obs["state"]["right/relative2_tcp_pose"], obs["state"]["right/relative2_tcp_vel"],
        # obs["state"]["right/wrist_tcp_pose"][t],
        obs["state"]["right/wrist_tcp_vel"],
        obs['state']['right/gripper_pos'],
    ], axis=-1)
    return torch.from_numpy(state).float()

def get_image(obs, camera_names, horizon=1, bgr=False):
    images_history = []
    for t in range(horizon):
        curr_images = []
        for cam_name in camera_names:
            curr_image = obs['images'][cam_name][t]
            if bgr:
                curr_image = curr_image[..., ::-1].copy()
            curr_images.append(torch.from_numpy(curr_image).permute(2, 0, 1))
        curr_images = torch.stack(curr_images, dim=0)
        images_history.append(curr_images)
    images_history = torch.stack(images_history, dim=0)
    return images_history.unsqueeze(0)

def get_image_batched(obs, camera_names, bgr=False):
    images = []
    for cam_name in camera_names:
        curr_image = obs['images'][cam_name] # B, T, H, W, C
        if bgr:
            curr_image = curr_image[..., ::-1].copy()
        images.append(torch.from_numpy(curr_image))
    images = torch.stack(images, dim=2) # B, T, K, C, H, W
    images = rearrange(images, "b t k h w c -> b t k c h w")
    return images

def save_video(frames, filename, fps=60):
    imageio.mimsave(filename, frames, fps=fps)
    print(f"Saved video to {filename}")

def save_video_texts_overlay(frames, filename, texts, fps=60):
    """
    Saves a video with an overlay text on each frame.

    Args:
        frames (list or array): A list (or array) of frames (numpy arrays).
        filename (str): Path where the video will be saved.
        texts (list): A list of strings to overlay on each frame.
                      Must have the same length as frames.
        fps (int): Frames per second for the video.
    """
    if len(frames) != len(texts):
        raise ValueError("The length of texts must match the number of frames.")

    processed_frames = []
    for frame, text in zip(frames, texts):
        frame_with_text = frame.copy()
        H, W = frame_with_text.shape[:2]
        position = (10, H - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 0, 0)  # white text (works for both BGR and RGB)
        thickness = 2
        line_type = cv2.LINE_AA

        # Overlay the text on the frame.
        cv2.putText(frame_with_text, text, position, font, font_scale, color, thickness, line_type)
        processed_frames.append(frame_with_text)

        # Save the processed frames as a video.
    imageio.mimsave(filename, processed_frames, fps=fps)
    print(f"Saved video to {filename}")

def video_writer_loop(
        frame_queue: queue.Queue, filename: str, fps: int, video_wh: tuple[int, int],
        image_aug_fn=None
    ):
    # e.g. fourcc='mp4v' for .mp4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, fps, video_wh)
    while True:
        item = frame_queue.get()
        if item is None: # sentinel signals end‐of‐stream
            break
        frames, text = item
        frames = torch.from_numpy(np.stack(frames, axis=0)) # (K, H, W, C)
        if image_aug_fn is not None:
            frames = image_aug_fn(rearrange(frames, "k h w c -> k c h w"))
            frames = rearrange(frames, "k c h w -> k h w c") # (K, H, W, C)
        frame = rearrange(frames, "k h w c -> h (k w) c").numpy() # (H, (K W), C)
        # overlay text in exactly the same way:
        position = (10, video_wh[1] - 10)
        frame = cv2.putText(frame, text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    print(f"Video writer thread finished writing to {filename}")

@torch.inference_mode()
def evaluate_policy(
    agent,
    make_env_fn,
    image_aug_fn,
    eval_episodes: int,
    num_envs: int,
    query_freq: int,
    horizon: int,
    act_dim: int,
    device: torch.device,
):
    assert eval_episodes % num_envs == 0, "eval_episodes should be divisible by num_envs"
    batch_num = eval_episodes // num_envs
    bar = tqdm(total=eval_episodes, desc="Eval Episodes", leave=True)

    vec_env = AsyncVectorEnv([make_env_fn() for _ in range(num_envs)])
    total_success = 0
    total_max_reward = 0

    for batch_idx in range(batch_num):
        actions_chunks = np.zeros((num_envs, horizon, act_dim), dtype=np.float32)
        max_rewards = np.zeros(num_envs, dtype=np.float32)
        t = 0
        obs, infos = vec_env.reset()
        active = np.ones(num_envs, dtype=bool)

        while np.any(active):
            if t % query_freq == 0:
                # Prepare batch for policy
                image_obs = torch.from_numpy(obs["pixels"]).unsqueeze(2).to(device) / 255.0
                image_obs = image_aug_fn(rearrange(image_obs, "b t k h w c -> b t k c h w"))
                state_obs = torch.from_numpy(obs["agent_pos"]).float().to(device)

                batch = {
                    "observation.image": image_obs.to(device, non_blocking=True),
                    "observation.state": state_obs.to(device, non_blocking=True),
                }

                actions, _ = agent.sample(batch)
                actions_chunks = actions.cpu().numpy()

            obs, rewards, dones, truncations, infos = vec_env.step(actions_chunks[:, t % query_freq])
            max_rewards = np.maximum(max_rewards, rewards)

            for i in range(num_envs):
                if active[i] and (dones[i] or truncations[i]):
                    active[i] = False
                    bar.update(1)

            t += 1

        total_success += np.sum(max_rewards == 1)
        total_max_reward += max_rewards.mean()

    bar.close()
    vec_env.close()
    success_rate = total_success / eval_episodes
    avg_max_reward = total_max_reward / batch_num

    print(f"[Eval] Success rate: {success_rate:.3f} ({total_success} / {eval_episodes})")
    print(f"[Eval] Average max reward: {avg_max_reward:.3f}")
    return {
        "success_rate": success_rate,
        "total_success": total_success,
        "average_max_reward": avg_max_reward,
    }

@torch.inference_mode()
def evaluate_bimanual_policy(
    agent,
    make_env_fn,
    image_aug_fn,
    eval_episodes: int,
    num_envs: int,
    query_freq: int,
    horizon: int,
    act_dim: int,
    device: torch.device,
    camera_names: list,
    time_limit: int,
):
    assert eval_episodes % num_envs == 0, "eval_episodes should be divisible by num_envs"
    batch_num = eval_episodes // num_envs
    bar = tqdm(total=eval_episodes, desc="Eval Episodes", leave=True)

    vec_env = AsyncVectorEnv([make_env_fn() for _ in range(num_envs)], context="spawn")
    total_successes = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }
    total_max_reward = 0

    for batch_idx in range(batch_num):
        actions_chunks = np.zeros((num_envs, horizon, act_dim), dtype=np.float32)
        max_rewards = np.zeros(num_envs, dtype=np.float32)
        t = 0
        obs, infos = vec_env.reset()
        active = np.ones(num_envs, dtype=bool)
        total_bar = tqdm(total=time_limit, desc="Env Steps", leave=True)

        while np.any(active):
            if t % query_freq == 0:
                # Prepare batch for policy
                state_obs = get_state_batched(obs).float()
                image_obs = image_aug_fn(
                    get_image_batched(obs, camera_names, bgr=False)
                )

                batch = {
                    "observation.image": image_obs.to(device, non_blocking=True).float() / 255.,
                    "observation.state": state_obs.to(device, non_blocking=True),
                }
                actions, _ = agent.sample(batch)
                actions_chunks = actions.cpu().numpy()

            obs, rewards, dones, truncations, infos = vec_env.step(actions_chunks[:, t % query_freq])
            max_rewards = np.maximum(max_rewards, rewards)

            for i in range(num_envs):
                if active[i] and (dones[i] or truncations[i]):
                    active[i] = False
                    bar.update(1)

            t += 1
            total_bar.update(1)

        for score in [0, 1, 2, 3, 4]:
            total_successes[score] += np.sum(max_rewards == score)
        total_max_reward += max_rewards.mean()
        total_bar.close()

    bar.close()
    vec_env.close()
    avg_max_reward = total_max_reward / batch_num

    for score in [0, 1, 2, 3, 4]:
        print(f"[Eval] Score {score}: {total_successes[score]} / {eval_episodes}")
    print(f"[Eval] Average max reward: {avg_max_reward:.3f}")
    return {
        "total_successes": total_successes,
        "average_max_reward": avg_max_reward,
    }