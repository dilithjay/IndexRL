import os
import pickle
import random
from glob import glob
from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np

from gpt import GPT, GPTConfig
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

SPLIT_COUNT = 100
MAX_EXP_LEN = 14

# Set device and seed
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed()

postfix = "-serp"
cache_dir = f"cache/cache{postfix}"
logs_dir = f"logs/logs{postfix}-of"
models_dir = f"models/models{postfix}-of"
for dir_name in (logs_dir, models_dir):
    Path(dir_name).mkdir(parents=True, exist_ok=True)


def main():
    with open(f"{logs_dir}/aucs.txt", "a") as _:
        pass
    with open(f"{logs_dir}/root_vals.txt", "a") as _:
        pass

    image_path = "data/images.npy"  # "data/10-0.npy"
    mask_path = "data/masks.npy"  # f"data/10-0{postfix}.npy"
    image = np.load(image_path)
    image = standardize(image)
    mask = np.load(mask_path)
    print(image.shape, mask.shape)

    n_channels = image.shape[0]
    action_list = list("()+-*/=") + ["sq", "sqrt"] + [f"c{c}" for c in range(n_channels)]

    data_buffer = []
    caches = sorted(glob(f"{cache_dir}/*.pkl"))
    if caches:
        with open(caches[-1], "rb") as fp:
            data_buffer = pickle.load(fp)

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

    n = 0
    models = sorted(glob(f"{models_dir}/*.pt"))
    pretrained = False

    # Define model
    if pretrained:
        if models:
            model_path = models[-1]
            n = int(os.path.basename(model_path).split("_")[1])
        else:
            model_path = "/home/dilith/Projects/IndexRL/models/models-pt-10c/model_15_loss-0.002919663543192049.pt"
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

    train_iterations = 100
    epochs_per_iter = 50
    all_losses = []
    for i in range(n + 1, n + train_iterations + 1):
        print(f"----------------\nIteration {i}")

        losses = []
        for _ in tqdm(range(epochs_per_iter), "Training..."):
            for state, acts in buffer:
                state = state.to(device)
                acts = acts.long().to(device)
                _, loss = agent(state, acts)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss = sum(losses) / len(losses)
        print(f"Loss: {loss}")
        all_losses.append(loss)

        i_str = str(i).rjust(2, "0")
        torch.save(agent, f"{models_dir}/model_{i_str}_loss-{loss}.pt")

        with open(f"{logs_dir}/losses.pkl", "wb") as fp:
            pickle.dump(losses, fp)


if __name__ == "__main__":
    main()
