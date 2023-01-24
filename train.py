import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataset import SatelliteDataset
from environment import IndexRLEnv
from agent import IndexRLAgent
from mcts import MCTS


def collect_data(env, agent, dataloader, image_count=None, n_iters=None):
    data = []
    for i, (image, mask) in enumerate(dataloader):
        if i == image_count:
            break
        
        done = False
        image_env = env.copy()
        mcts = MCTS(image_env.copy(), agent, image, mask, True)
        state = image_env.reset(image, mask)
        while not done:
            probs = mcts.run(n_iters) if n_iters else mcts.run()
            data.append((state, probs))
            action = random.choices(range(len(probs)), weights=probs, k=1)[0]
            state, _, done = image_env.step(action)
            mcts = MCTS(image_env.copy(), agent, image, mask)
        print(image_env.cur_exp)
            
    return data


def main():
    max_exp_len = 7

    dataset = SatelliteDataset('D:/FYP/Code/SerpSeg/data/train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    image, _ = dataset[0]
    n_channels = image.shape[0]

    n_channels = image.size(0)
    action_list = list("()+-*/=") + ['sq', 'sqrt'] + [f"c{c}" for c in range(n_channels)]

    action_size = len(action_list)
    state_size = image.flatten().size(0) + max_exp_len

    main_env = IndexRLEnv(action_list, max_exp_len)
    agent = IndexRLAgent(action_size, state_size)
    agent = agent.float()
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(agent.parameters(), )
    
    train_iterations = 10
    epochs_per_iter = 10
    for i in range(1, train_iterations + 1):
        print(f"----------------\nIteration {i}")
        print("Collecting data...")
        data = collect_data(main_env.copy(), agent, dataloader, image_count=1, n_iters=(i + max_exp_len) * action_size)
        print("Data collection done.")
        for _ in tqdm(range(epochs_per_iter), "Training..."):
            for state, action_probs in data:
                pred = agent(torch.tensor(state).float())
                loss = criterion(pred, torch.tensor(action_probs).float())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.zero_grad()


if __name__ == "__main__":
    main()
