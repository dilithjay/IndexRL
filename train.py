import os
import random
from glob import glob
from tqdm import tqdm

import torch
import numpy as np

from environment import IndexRLEnv
from agent import IndexRLAgent
from mcts import MCTS
from utils import set_seed, standardize

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed()


def collect_data(env, agent, image, mask, n_iters=None):
    data = []

    split_count = 10
    split_size = len(mask) // split_count
    for i in range(split_count):
        done = False
        count = 0
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
            state, _, done = image_env.step(action)
            data.append((state, probs))
            tree_str = mcts.root_node.display_tree(stdout=False)
            with open(f"logs/tree_{count}.txt", "w") as fp:
                fp.write(tree_str)
            mcts = MCTS(image_env.copy(), agent, img_split, mask_split)
        with open("logs/aucs.txt", "a") as fp:
            fp.write(f"{i} {image_env.best_reward} {image_env.best_exp}\n")

        print(image_env.cur_exp)

    return data


def main():
    with open("logs/aucs.txt", "w") as _:
        pass

    max_exp_len = 12
    image = np.load("data/images.npy")
    image = standardize(image)
    mask = np.load("data/masks.npy")

    n_channels = image.shape[0]
    action_list = list("()+-*/=") + ["sq", "sqrt"] + [f"c{c}" for c in range(n_channels)]

    action_size = len(action_list)
    state_size = max_exp_len * action_size

    main_env = IndexRLEnv(action_list, max_exp_len)
    n = 0
    models = sorted(glob("models/*.pt"))
    pretrained = True

    if pretrained and models:
        model_path = models[-1]
        print(model_path)
        n = int(os.path.basename(model_path).split("_")[1])
        agent = torch.load(model_path)
    else:
        agent = IndexRLAgent(action_size, state_size)
        agent = agent.float()
    agent.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        agent.parameters(),
    )

    train_iterations = 20
    epochs_per_iter = 20
    for i in range(n + 1, n + train_iterations + 1):
        print(f"----------------\nIteration {i}")
        print("Collecting data...")
        # n_iters = (i + max_exp_len) * action_size * 2
        data = collect_data(main_env.copy(), agent, image, mask, n_iters=500)
        print("Data collection done.")
        losses = []
        for _ in tqdm(range(epochs_per_iter), "Training..."):
            for state, action_probs in data:
                pred = agent(torch.tensor(state).to(device).float())
                loss = criterion(pred, torch.tensor(action_probs).float().to(device))
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss = sum(losses) / len(losses)
        print(f"Loss: {loss}")
        torch.save(agent, f"models/model_{i}_loss-{loss}.pt")


if __name__ == "__main__":
    main()
