import gym
from gym import spaces

import numpy as np

from expression_handler import eval_expression


class IndexRLEnv(gym.Env):
    def __init__(self, discrete_actions: list, image: np.ndarray, mask: np.ndarray, max_exp_len: int):
        super(IndexRLEnv, self).__init__()
        self.actions = discrete_actions
        self.image = image
        self.mask = mask
        self.cur_exp = []
        self.max_exp_len = max_exp_len

    def step(self, action_tpl: tuple):
        done = 0

        done = self.take_action(action_tpl)

        if len(self.cur_exp) >= self.max_exp_len:
            done = 1

        reward = self.get_reward(done)

        return self.cur_exp, reward, done

    def reset(self):
        self.cur_exp = []

    def render(self):
        pass

    def get_reward(self, done: bool):
        result = eval_expression(self.cur_exp, self.image)
        if result is False:
            if done:
                return -10
            return -1
        if done:
            return get_auc_precision_recall(result, self.mask) * 40

    def take_action(self, action_tpl):
        action = self.actions[action_tpl[0]]
        # If action one of => opening brackets / constant / pow / channel
        if (action in "(1p") or ("c" in action):
            self.cur_exp += [action_tpl[1], action]
        else:
            self.cur_exp.append(action)

        return action == "="


def get_precision_recall(result: np.ndarray, mask: np.ndarray, threshold: float = 0.5):
    pred_mask = result > threshold
    tp = np.logical_and(pred_mask, mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(mask)).sum()
    tn = np.logical_and(np.logical_not(pred_mask), np.logical_not(mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), mask).sum()

    return tp / (tp + fp), tp / (tp + fn)


def get_auc_precision_recall(result, mask):
    tot_pr = 0
    for thresh in np.arange(0.1, 1, 0.1):
        tot_pr += get_precision_recall(result, mask, thresh)

    return tot_pr / 9


def main():
    n_channels = 3
    action_list = list("()+*p1=") + [f"c{c}" for c in range(n_channels)]
    image = np.random.rand(n_channels, 5, 5)
    mask = np.random.rand(n_channels, 5, 5)
    env = IndexRLEnv(action_list, image, mask, 100)
