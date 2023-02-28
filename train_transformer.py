import os
import pickle
import random
from glob import glob
from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np

from environment import IndexRLEnv
from gpt import GPT, GPTConfig
from mcts import MCTS
from utils import set_seed, standardize
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

# Set device and seed
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed()

postfix = "-serp"
cache_dir = f"cache/cache{postfix}"
logs_dir = f"logs/logs{postfix}"
models_dir = f"models/models{postfix}"
for dir_name in (cache_dir, logs_dir, models_dir):
    Path(dir_name).mkdir(parents=True, exist_ok=True)


def collect_data(env, agent, image, mask, n_iters=None):
    data = []

    split_count = 100
    split_size = len(mask) // split_count
    root_vals = []

    for i in range(split_count):
        print("Split:", i)

        done = False
        count = 0
        root_vals_split = []

        image_env = env.copy()
        img_split = image[:, split_size * i : split_size * (i + 1)]
        mask_split = mask[split_size * i : split_size * (i + 1)]
        mcts = MCTS(image_env.copy(), agent, img_split, mask_split, True)
        state = image_env.reset(img_split, mask_split)

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

        data.append((state, probs, reward))

        with open(f"{logs_dir}/aucs.txt", "a") as fp:
            fp.write(f"{i} {image_env.best_reward} {image_env.best_exp}\n")

        root_vals.append(root_vals_split)
        print(image_env.cur_exp)

    with open(f"{logs_dir}/root_vals.txt", "a") as fp:
        fp.write(f"{np.concatenate(root_vals).mean()}\t{root_vals}\n")

    return data


def main():
    with open(f"{logs_dir}/aucs.txt", "a") as _:
        pass
    with open(f"{logs_dir}/root_vals.txt", "a") as _:
        pass

    max_exp_len = 14
    image_path = "data/images.npy"  # "data/10-0.npy"
    mask_path = "data/masks.npy"  # f"data/10-0{postfix}.npy"
    image = np.load(image_path)
    image = standardize(image)
    mask = np.load(mask_path)
    print(image.shape, mask.shape)

    n_channels = image.shape[0]
    action_list = list("()+-*/=") + ["sq", "sqrt"] + [f"c{c}" for c in range(n_channels)]

    main_env = IndexRLEnv(action_list, max_exp_len, False, masked_actions=["sq", "sqrt"])

    data_buffer = []
    caches = sorted(glob(f"{cache_dir}/*.pkl"))
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
            model_path = "models/models-pt-2/model_23_loss-8.56757884549163.pt"
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

    max_buffer_size = 4000
    train_iterations = 100
    epochs_per_iter = 30
    for i in range(n + 1, n + train_iterations + 1):
        print(f"----------------\nIteration {i}")
        print("Collecting data...")
        # n_iters = (i + max_exp_len) * action_size * 2
        data = collect_data(main_env.copy(), agent, image, mask, n_iters=1000)
        data_buffer = (data + data_buffer)[:max_buffer_size]
        buffer_cp = data_buffer.copy()
        random.shuffle(buffer_cp)
        print(f"Data collection done. Collected {len(data)} examples. Buffer size = {len(data_buffer)}")

        losses = []
        for _ in tqdm(range(epochs_per_iter), "Training..."):
            for state, probs, reward in data:
                state = torch.tensor(np.expand_dims(state, axis=0)).int().to(device)
                probs = torch.tensor(np.expand_dims(probs, axis=0)).float().to(device) * 100
                _, loss = agent(state, probs)
                losses.append((loss * (1 - reward)).item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss = sum(losses) / len(losses)
        print(f"Loss: {loss}")
        torch.save(agent, f"{models_dir}/model_{i}_loss-{loss}.pt")

        with open(f"{cache_dir}/data_buffer_{i}.pkl", "wb") as fp:
            pickle.dump(data_buffer, fp)


if __name__ == "__main__":
    main()
