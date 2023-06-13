import os
from pathlib import Path
import pickle
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

# Set device and seed
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed()

postfix = "-pt-13c"
cache_dir = f"cache/cache{postfix}"
logs_dir = f"logs/logs{postfix}"
models_dir = f"models/models{postfix}"
for dir_name in (cache_dir, logs_dir, models_dir):
    Path(dir_name).mkdir(parents=True, exist_ok=True)


def collect_data(env, agent, image, mask, n_iters=None):
    data = []

    split_count = 5
    root_vals = []

    for i in range(split_count):
        print("Split:", i)

        done = False
        count = 0
        root_vals_split = []

        image_env = env.copy()
        state = image_env.reset(image, mask)
        # image_env.step()
        mcts = MCTS(image_env.copy(), agent, image, mask, True)

        while not done:
            count += 1

            probs = mcts.run(max(n_iters // 2, n_iters - 50 * count)) if n_iters else mcts.run()
            probs[0] *= 5  # increase prob of predicting "(" during pretraining
            action = random.choices(range(len(probs)), weights=probs, k=1)[0]
            state, reward, done = image_env.step(action)

            root_vals_split.append(round(mcts.root_node.value / mcts.root_node.n, 4))
            tree_str = mcts.root_node.display_tree(stdout=False)
            with open(f"{logs_dir}/tree_{count}.txt", "w") as fp:
                fp.write(f"Expression: {mcts.env.cur_exp}\n{tree_str}")

            mcts = MCTS(image_env.copy(), agent, image, mask)

        root_vals.append(root_vals_split)
        print(image_env.cur_exp)

        if reward > 1:
            data.append((state, reward))

    with open(f"{logs_dir}/root_vals.txt", "a") as fp:
        fp.write(f"{np.concatenate(root_vals).mean()}\t{root_vals}\n")

    return data


def main():

    max_exp_len = 14
    image = np.ones((13, 2, 2))
    mask = np.ones((2, 2))
    print(image.shape, mask.shape)

    n_channels = image.shape[0]
    action_list = list("()+-*/=") + ["sq", "sqrt"] + [f"c{c}" for c in range(n_channels)]

    main_env = IndexRLEnv(action_list, max_exp_len, False, masked_actions=["sq", "sqrt"], unitless=True)

    n = 0
    models = sorted(glob(f"{models_dir}/*.pt"))
    pretrained = True

    # Define model
    if pretrained and models:
        model_path = models[-1]
        print(model_path)
        n = int(os.path.basename(model_path).split("_")[1])
        agent = torch.load(model_path)
    else:
        with open(f"{logs_dir}/root_vals.txt", "w") as _:
            pass
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

    data_buffer = []
    caches = sorted(glob(f"{cache_dir}/*.pkl"))
    if caches:
        with open(caches[-1], "rb") as fp:
            data_buffer = pickle.load(fp)

    max_buffer_size = 2000
    train_iterations = 10
    epochs_per_iter = 30
    for i in range(n + 1, n + train_iterations + 1):
        print(f"----------------\nIteration {i}")
        print("Collecting data...")
        # n_iters = (i + max_exp_len) * action_size * 2
        data = collect_data(main_env.copy(), agent, image, mask, n_iters=1000)
        data_buffer = sorted(data + data_buffer, key=lambda x: x[1], reverse=True)[:max_buffer_size]
        print(f"Data collection done. Collected {len(data)} examples. Total length: {len(data_buffer)}")

        states = {}
        actions = {}
        for state, _ in data_buffer:
            states[len(state) - 1] = states.get(len(state) - 1, []) + [state[:-1]]
            actions[len(state) - 1] = actions.get(len(state) - 1, []) + [state[1:]]

        buffer = []
        for key in states:
            state = torch.tensor(np.array(states[key]))
            acts = torch.tensor(np.array(actions[key]))
            buffer.append((state, acts))
        random.shuffle(buffer)

        losses = []
        for _ in tqdm(range(epochs_per_iter), "Training..."):
            for state, acts in buffer:
                state = state.to(device)
                acts = acts.to(device)
                _, loss = agent(state, acts)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss = sum(losses) / len(losses)
        print(f"Loss: {loss}")

        i_str = str(i).rjust(2, "0")
        torch.save(agent, f"{models_dir}/model_{i_str}_loss-{loss}.pt")

        with open(f"{cache_dir}/data_buffer_{i_str}.pkl", "wb") as fp:
            pickle.dump(data_buffer, fp)


if __name__ == "__main__":
    main()
