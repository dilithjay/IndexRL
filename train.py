import random
from tqdm import tqdm

import torch
import numpy as np

from environment import IndexRLEnv
from agent import IndexRLAgent
from mcts import MCTS
from utils import set_seed

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed()


def collect_data(env, agent, image, mask, n_iters=None):
    data = []
    done = False
    count = 0

    image_env = env.copy()
    mcts = MCTS(image_env.copy(), agent, image, mask, True)
    state = image_env.reset(image, mask)

    while not done:
        count += 1
        probs = mcts.run(n_iters) if n_iters else mcts.run()
        action = random.choices(range(len(probs)), weights=probs, k=1)[0]
        state, _, done = image_env.step(action)
        if count % 100 == 0:
            data.append((state, probs))
        tree_str = mcts.root_node.display_tree(stdout=False)
        with open(f"logs/tree_{count}.txt", "w") as fp:
            fp.write(tree_str)
        mcts = MCTS(image_env.copy(), agent, image, mask)

    print(image_env.cur_exp)
    with open("logs/aucs.txt", "a") as fp:
        fp.write(f"{image_env.max_auc_seen}\n")

    return data


def main():
    max_exp_len = 12
    image = np.load("data/images.npy")
    image = (image - image.min(axis=1)[:, None]) / (image.max(axis=1)[:, None] - image.min(axis=1)[:, None])
    mask = np.load("data/masks.npy")

    n_channels = image.shape[0]
    action_list = list("()+-*/=") + ["sq", "sqrt"] + [f"c{c}" for c in range(n_channels)]

    action_size = len(action_list)
    state_size = max_exp_len * action_size

    main_env = IndexRLEnv(action_list, max_exp_len)
    agent = IndexRLAgent(action_size, state_size)
    agent = agent.float()
    agent.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        agent.parameters(),
    )

    train_iterations = 20
    epochs_per_iter = 20
    for i in range(1, train_iterations + 1):
        print(f"----------------\nIteration {i}")
        print("Collecting data...")
        # n_iters = (i + max_exp_len) * action_size * 2
        data = collect_data(main_env.copy(), agent, image, mask, n_iters=500)
        print("Data collection done.")
        for _ in tqdm(range(epochs_per_iter), "Training..."):
            for state, action_probs in data:
                pred = agent(torch.tensor(state).to(device).float())
                loss = criterion(pred, torch.tensor(action_probs).float().to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


if __name__ == "__main__":
    main()
