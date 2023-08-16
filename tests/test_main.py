import os
import pickle
import torch
from glob import glob
from pathlib import Path

from indexrl.training import (
    DynamicBuffer,
    create_model,
    save_model,
    explore,
    train_iter,
)
from indexrl.environment import IndexRLEnv
from indexrl.utils import get_n_channels, set_seed

set_seed(0)

data_dir = "tests/data/"

image_dir = os.path.join(data_dir, "images")
mask_dir = os.path.join(data_dir, "masks")

img_path = glob(os.path.join(image_dir, "*.npy"))[0]
n_channels = get_n_channels(img_path)

cache_dir = os.path.join(data_dir, "cache")
logs_dir = os.path.join(data_dir, "logs")
models_dir = os.path.join(data_dir, "models")
for dir_name in (cache_dir, logs_dir, models_dir):
    Path(dir_name).mkdir(parents=True, exist_ok=True)

action_list = list("()+-*/=") + ["sq", "sqrt"] + [f"c{c}" for c in range(n_channels)]


def initialize_objects():
    env = IndexRLEnv(action_list, 12)
    agent, optimizer = create_model(len(action_list))
    seen_path = os.path.join(cache_dir, "seen.pkl") if cache_dir else ""
    env.save_seen(seen_path)
    data_buffer = DynamicBuffer()

    return env, agent, optimizer, data_buffer, seen_path


def test_object_initialization():
    initialize_objects()


def test_explore():
    env, agent, _, _, seen_path = initialize_objects()

    data = explore(
        env.copy(),
        agent,
        image_dir,
        mask_dir,
        1,
        logs_dir,
        seen_path,
        n_iters=1000,
    )

    assert len(data) == 1


def test_data_buffer_add():
    env, agent, _, data_buffer, seen_path = initialize_objects()

    for i in range(3):
        data = explore(
            env.copy(),
            agent,
            image_dir,
            mask_dir,
            1,
            logs_dir,
            seen_path,
            n_iters=1000,
        )

        data_buffer.add_data(data)

        assert len(data_buffer) == i + 1


def test_find_expression():
    env, agent, optimizer, data_buffer, seen_path = initialize_objects()

    for i in range(2):
        data = explore(
            env.copy(),
            agent,
            image_dir,
            mask_dir,
            1,
            logs_dir,
            seen_path,
            n_iters=1000,
        )

        data_buffer.add_data(data)

        agent, optimizer, loss = train_iter(agent, optimizer, data_buffer)
        assert not torch.isnan(torch.tensor(loss))

        i_str = str(i).rjust(3, "0")
        if models_dir:
            save_model(agent, f"{models_dir}/model_{i_str}_loss-{loss}.pt")
        if cache_dir:
            with open(f"{cache_dir}/data_buffer_{i_str}.pkl", "wb") as fp:
                pickle.dump(data_buffer, fp)
