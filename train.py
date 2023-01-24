import torch
from torch.utils.data import DataLoader

from dataset import SatelliteDataset
from environment import IndexRLEnv
from agent import IndexRLAgent
from mcts import MCTS


def collect_data(env, agent, dataloader, image_count):
    data = []
    for i, (image, mask) in enumerate(dataloader):
        if i == image_count:
            break
        
        mcts = MCTS(env, agent, image, mask)
        start_state = mcts.root_node.state
        probs = mcts.run(20)
        data.append((start_state, probs))
    
    return data


def main():
    max_exp_len = 20

    dataset = SatelliteDataset('D:/FYP/Code/SerpSeg/data/train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    image, _ = dataset[0]
    n_channels = image.shape[0]

    n_channels = image.size(0)
    action_list = list("()+-*\=") + ['sq', 'sqrt'] + [f"c{c}" for c in range(n_channels)]

    action_size = len(action_list)
    state_size = image.flatten().size(0) + max_exp_len

    main_env = IndexRLEnv(action_list, max_exp_len)
    agent = IndexRLAgent(action_size, state_size)
    agent = agent.float()
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(agent.parameters(), )
    
    epochs = 10
    for ep in range(epochs):
        data = collect_data(main_env.copy(), agent, dataloader, 3)
        for state, action_probs in data:
            pred = agent(torch.tensor(state).float())
            loss = criterion(pred, torch.tensor(action_probs).float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.zero_grad()


if __name__ == "__main__":
    main()
