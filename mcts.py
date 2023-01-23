import math
import copy
import numpy as np
import torch

from tqdm import tqdm
from treelib import Tree

from environment import IndexRLEnv
from agent import IndexRLAgent
from dataset import SatelliteDataset
from utils import set_seed


class Node():
    def __init__(self, state=None, name="", index=0):
        self.value = 0
        self.n = 0
        self.state = state
        self.name = name
        self.index = index
        self.children = []
    
    def get_max_ucb1_child(self):
        if not self.children:
            return None
        
        max_node = self.children[0]
        max_ucb1 = float('-inf')
        
        for child in self.children:
            ucb1 = self.get_child_ucb1(child)
            
            if ucb1 > max_ucb1:
                max_ucb1 = ucb1
                max_node = child
        
        return max_node
    
    def get_child_ucb1(self, child):
        if child.n == 0:
            return float('inf')
        return child.value / child.n + 2 * math.sqrt(math.log(self.n, math.e) / child.n)
    
    def display_tree(self):
        tree = Tree()

        tree.create_node(f"Root => value: {self.value}, n: {self.n}", "root")
        stack = [(self, 'root')]
        count = 0
        
        while stack:
            cur_node, node_name = stack.pop()
            for child in cur_node.children:
                tree.create_node(f"{child.name} => value: {child.value}, n: {child.n}", f"node_{count}", parent=node_name)
                stack.append((child, f"node_{count}"))
                count += 1

        tree.show()


class MCTS():
    def __init__(self, env, agent, image, mask):
        self.env = env
        self.agent = agent
        
        start_state = self.env.reset(image, mask)
        self.root_node = Node(start_state)
    
    def run(self, n_iter=200):
        for act in self.env.get_valid_actions():
            env_copy = copy.deepcopy(self.env)
            new_state, _, _ = env_copy.step(act)
            new_node = Node(new_state, self.env.actions[act], act)
            self.root_node.children.append(new_node)
        self.root_node.display_tree()
        for _ in tqdm(range(n_iter)):
            value, node_path = self.traverse()
            self.backpropagate(node_path, value)
            self.env.reset()
        
        self.root_node.display_tree()
        vals = [0] * len(self.env.actions)
        for child in self.root_node.children:
            vals[child.index] = child.value / child.n
        
        return np.exp(vals) / sum(np.exp(vals))
        
    def traverse(self):
        cur_node = self.root_node
        node_path = [cur_node]
        while cur_node.children:
            cur_node = cur_node.get_max_ucb1_child()
            self.env.step(cur_node.index)
            node_path.append(cur_node)
        if cur_node.n:
            for act in self.env.get_valid_actions():
                env_copy = copy.deepcopy(self.env)
                new_state, _, _ = env_copy.step(act)
                new_node = Node(new_state, self.env.actions[act])
                cur_node.children.append(new_node)
            while cur_node.children:
                cur_node = cur_node.get_max_ucb1_child()
                node_path.append(cur_node)
        # print("Node path:", [node.name for node in node_path])
        
        return self.rollout(cur_node), node_path
    
    def backpropagate(self, node_path: list, last_value: float):
        for node in node_path[::-1]:
            node.value += last_value
            node.n += 1
    
    def rollout(self, state_node: Node) -> float:
        tot_reward = 0
        cur_state = state_node.state
        
        step = 0
        while True:
            # print(step, tot_reward, self.env.cur_exp)
            step += 1
            probs = self.agent(torch.tensor(cur_state).float())
            
            invalid_acts = self.env.get_invalid_actions()
            probs[invalid_acts] = -1
            
            action_idx = probs.argmax()
            
            cur_state, reward, done = self.env.step(action_idx)
            tot_reward += reward
            
            if done:
                break
        
        return tot_reward
    

def main():
    n_channels = 10
    max_exp_len = 20
    
    dataset = SatelliteDataset('D:/FYP/Code/SerpSeg/data/train', 'D:/FYP/Code/SerpSeg/data/train')
    image, mask = dataset[0]
    
    n_channels = image.size(0)
    action_list = list("()+-*\=") + ['sq', 'sqrt'] + [f"c{c}" for c in range(n_channels)]
    
    action_size = len(action_list)
    state_size = image.flatten().size(0) + max_exp_len
    
    env = IndexRLEnv(action_list, max_exp_len)
    agent = IndexRLAgent(action_size, state_size)
    agent = agent.float()
    
    mcts = MCTS(env, agent, image, mask)
    probs = mcts.run()
    print(probs)


if __name__ == "__main__":
    set_seed(42)
    main()
        