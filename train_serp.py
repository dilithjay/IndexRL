import os
import pickle
import random
from glob import glob
from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np

from environment import IndexRLEnv, get_final_reward
from gpt import GPT, GPTConfig
from mcts import MCTS
from utils import set_seed
from configs.config_transfomer import (
    n_layer,
    n_head,
    n_embd,
    block_size,
    bias,
    dropout,
    weight_decay,
    learning_rate,
    beta1,
    beta2,
)

MAX_EXP_LEN = 14

# Set device and seed
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(1)

image_dir = "/home/dilith/Projects/SerpSeg/data-serp/images/"
mask_dir = "/home/dilith/Projects/SerpSeg/data-serp/masks/"
image_paths = glob(os.path.join(image_dir, "*.npy"))
mask_paths = glob(os.path.join(mask_dir, "*.npy"))

postfix = "-serp"
cache_dir = f"cache/cache{postfix}"
logs_dir = f"logs/logs{postfix}"
models_dir = f"models/models{postfix}"
for dir_name in (cache_dir, logs_dir, models_dir):
    Path(dir_name).mkdir(parents=True, exist_ok=True)

seen_path = os.path.join(cache_dir, "seen.pkl")
if not os.path.exists(seen_path):
    with open(seen_path, "wb") as fp:
        pickle.dump(set(), fp)


def collect_data(env, agent, n_iters=None):
    data = []
    root_vals = []
    max_splits = 5

    for i in range(max_splits):
        idx = random.randrange(0, len(image_paths) - 1)
        print("Split:", i, ", Image:", idx)
        image_path = image_paths[idx]
        mask_path = mask_paths[idx]

        done = False
        count = 0
        root_vals_split = []

        image_env = env.copy()
        img_split = np.load(image_path)
        mask_split = np.load(mask_path)
        mcts = MCTS(image_env.copy(), agent, img_split, mask_split, True)
        state = image_env.reset(img_split, mask_split)
        image_env.load_seen(seen_path)

        while not done:
            count += 1

            probs = mcts.run(max(n_iters // 2, n_iters - 50 * count)) if n_iters else mcts.run()
            action = random.choices(range(len(probs)), weights=probs, k=1)[0]
            state, reward, done = image_env.step(action)

            root_vals_split.append(round(mcts.root_node.value / mcts.root_node.n, 4))
            tree_str = mcts.root_node.display_tree(stdout=False)
            with open(f"{logs_dir}/tree_{count}.txt", "w") as fp:
                fp.write(f"Expression: {mcts.env.cur_exp}\n{tree_str}")
            mcts = MCTS(image_env.copy(), agent, img_split, mask_split)

        image_env.save_seen(seen_path)
        if reward > 0:
            print("Reward < 0. Skipping.")
            continue
        final_reward = get_final_reward(image_env.cur_exp, image_dir, mask_dir)
        if final_reward > 0.01:
            data.append((state, final_reward))

        with open(f"{logs_dir}/aucs.txt", "a") as fp:
            fp.write(f"{i} {final_reward} {image_env.cur_exp}\n")

        root_vals.append(root_vals_split)
        print(image_env.cur_exp, final_reward)

    if root_vals:
        with open(f"{logs_dir}/root_vals.txt", "a") as fp:
            fp.write(f"{np.concatenate(root_vals).mean()}\t{root_vals}\n")

    return data


def main():
    with open(f"{logs_dir}/aucs.txt", "a") as _:
        pass
    with open(f"{logs_dir}/root_vals.txt", "a") as _:
        pass

    n_channels = 10
    action_list = list("()+-*/=") + ["sq", "sqrt"] + [f"c{c}" for c in range(n_channels)]

    main_env = IndexRLEnv(action_list, MAX_EXP_LEN, False, masked_actions=["sq", "sqrt"], unitless=False)

    data_buffer = []
    caches = sorted(glob(f"{cache_dir}/data_buffer_*.pkl"))
    if caches:
        with open(caches[-1], "rb") as fp:
            data_buffer = pickle.load(fp)

    n = 0
    models = sorted(glob(f"{models_dir}/*.pt"))
    pretrained = True

    # Define model
    if pretrained:
        if models:
            model_path = models[-1]
            n = int(os.path.basename(model_path).split("_")[1])
        else:
            model_path = "models/models-serp-of/best_model.pt"
            n = 0
        print("Model path:", model_path)
        agent = torch.load(model_path)
    else:
        model_args = dict(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            bias=bias,
            vocab_size=len(action_list),
            dropout=dropout,
        )
        gptconf = GPTConfig(**model_args)
        agent = GPT(gptconf)
    agent.to(device)

    optimizer = agent.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)

    max_buffer_size = 500
    min_buffer_size = 20
    keep_fraction = 0.97
    epochs_per_iter = 100
    i = n
    max_reached = True
    while True:
        i += 1
        print(f"----------------\nIteration {i}")
        print("Collecting data...")
        data = collect_data(main_env.copy(), agent, n_iters=1000)
        print(f"Data collection done. Collected {len(data)} examples. Buffer size = {len(data_buffer)}.")

        # Dynamic buffer
        current_capacity = len(data_buffer)
        if not max_reached and len(data) + current_capacity < max_buffer_size:
            data_buffer = data + data_buffer
            continue
        max_reached = True
        if max_reached:
            data_buffer = sorted(data + data_buffer, key=lambda x: x[1], reverse=True)[
                : max(int(current_capacity * keep_fraction), min_buffer_size)
            ]
        print(f"Buffer size new = {len(data_buffer)}.")

        states = {}
        actions = {}
        rewards = {}
        reward_min, reward_max = 1, 0
        for state, reward in sorted(data_buffer, key=lambda x: x[1], reverse=True)[:max_buffer_size]:
            if len(state) < 4:
                continue
            states[len(state) - 1] = states.get(len(state) - 1, []) + [state[:-1]]
            actions[len(state) - 1] = actions.get(len(state) - 1, []) + [state[1:]]
            rewards[len(state) - 1] = rewards.get(len(state) - 1, []) + [reward]
            reward_min = min(reward_min, reward)
            reward_max = max(reward_max, reward)

        buffer = []
        for key in states:
            state = torch.tensor(np.array(states[key]))
            acts = torch.tensor(np.array(actions[key]))
            rews = torch.tensor(np.array(rewards[key]))
            buffer.append((state, acts, rews))
        random.shuffle(buffer)

        losses = None
        for _ in tqdm(range(epochs_per_iter), "Training..."):
            losses = []
            for state, acts, rews in buffer:
                state = state.to(device)
                acts = acts.to(device)
                _, loss = agent(state, acts)
                rew_scaled = (rews - reward_min) / (reward_max - reward_min)
                loss *= rew_scaled.mean()
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss = sum(losses) / len(losses)
        print(f"Loss: {loss}")

        i_str = str(i).rjust(3, "0")
        torch.save(agent, f"{models_dir}/model_{i_str}_loss-{loss}.pt")

        with open(f"{cache_dir}/data_buffer_{i_str}.pkl", "wb") as fp:
            pickle.dump(data_buffer, fp)


if __name__ == "__main__":
    main()
