import os
import random
from glob import glob
from tqdm import tqdm

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

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed()


def collect_data(env, agent, image, mask, n_iters=None):
    data = []

    split_count = 10
    split_size = len(mask) // split_count
    root_vals = []
    for i in range(split_count):
        done = False
        count = 0
        root_vals_split = []
        print("Split:", i)
        image_env = env.copy()
        img_split = image[:, split_size * i : split_size * (i + 1)]
        mask_split = mask[split_size * i : split_size * (i + 1)]
        mcts = MCTS(image_env.copy(), agent, img_split, mask_split, True)
        state = image_env.reset(img_split, mask_split)
        while not done:
            count += 1
            probs = mcts.run(n_iters) if n_iters else mcts.run()
            action = random.choices(range(len(probs)), weights=probs, k=1)[0]
            state, reward, done = image_env.step(action)
            data.append((state, probs, reward))
            root_vals_split.append(round(mcts.root_node.value / mcts.root_node.n, 4))
            tree_str = mcts.root_node.display_tree(stdout=False)
            with open(f"logs/tree_{count}.txt", "w") as fp:
                fp.write(tree_str)
            mcts = MCTS(image_env.copy(), agent, img_split, mask_split)
        with open("logs/aucs.txt", "a") as fp:
            fp.write(f"{i} {image_env.best_auc} {image_env.best_exp}\n")
        print(image_env.cur_exp)
        root_vals.append(root_vals_split)

    with open("logs/root_vals.txt", "a") as fp:
        fp.write(f"{np.concatenate(root_vals).mean()}\t{root_vals}\n")

    return data


def main():
    with open("logs/aucs.txt", "w") as _:
        pass
    with open("logs/root_vals.txt", "w") as _:
        pass

    max_exp_len = 12
    image = np.load("data/images.npy")
    image = standardize(image)
    mask = np.load("data/masks.npy")
    print(image.shape, mask.shape)

    n_channels = image.shape[0]
    action_list = list("()+-*/=") + ["sq", "sqrt"] + [f"c{c}" for c in range(n_channels)]

    main_env = IndexRLEnv(action_list, max_exp_len, False)

    n = 0
    models = sorted(glob("models/*.pt"))
    pretrained = True

    if pretrained and models:
        model_path = models[-1]
        print(model_path)
        n = int(os.path.basename(model_path).split("_")[1])
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

    train_iterations = 100
    epochs_per_iter = 30
    for i in range(n + 1, n + train_iterations + 1):
        print(f"----------------\nIteration {i}")
        print("Collecting data...")
        # n_iters = (i + max_exp_len) * action_size * 2
        data = collect_data(main_env.copy(), agent, image, mask, n_iters=500)
        print("Data collection done.")
        losses = []
        for _ in tqdm(range(epochs_per_iter), "Training..."):
            for state, probs, reward in data:
                state = torch.tensor(np.expand_dims(state, axis=0)).int().to(device)
                probs = torch.tensor(np.expand_dims(probs, axis=0)).float().to(device)
                _, loss = agent(state, probs)
                losses.append((loss * (1 - reward)).item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss = sum(losses) / len(losses)
        print(f"Loss: {loss}")
        torch.save(agent, f"models/model_{i}_loss-{loss}.pt")


if __name__ == "__main__":
    main()
