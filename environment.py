import gym
from copy import deepcopy
import numpy as np

from expression_handler import eval_expression


class IndexRLEnv(gym.Env):
    def __init__(self, discrete_actions: list, max_exp_len: int = 100):
        super(IndexRLEnv, self).__init__()
        self.actions = discrete_actions
        self.image = None
        self.mask = None
        self.cur_exp = []
        self.parentheses_level = 0
        self.max_exp_len = max_exp_len
        self.max_auc_seen = 0

    def get_cur_state(self):
        cur_exp_indices = [self.actions.index(act) for act in self.cur_exp] + [0] * (
            self.max_exp_len - len(self.cur_exp)
        )
        return np.concatenate([self.image.flatten(), cur_exp_indices])

    def step(self, action_idx: int) -> tuple:
        """Take a step in the environment with the specified action.

        Args:
            action_idx (int): discrete action index

        Returns:
            np.ndarray: current state
            float:      reward
            bool:       done
        """
        done = self.take_action(action_idx)

        if len(self.cur_exp) >= self.max_exp_len:
            done = True

        reward = self.get_reward(done)

        return self.get_cur_state(), reward, done

    def reset(self, image: np.ndarray = None, mask: np.ndarray = None) -> np.ndarray:
        if image is not None and mask is not None:
            self.image = image
            self.mask = mask
        self.cur_exp = []
        self.parentheses_level = 0

        return self.get_cur_state()

    def render(self):
        print(self.cur_exp)

    def get_reward(self, done: bool) -> float:
        result = eval_expression(self.cur_exp, self.image.squeeze())
        if result is False:
            if done:
                return -10
            return 0
        if done:
            auc = get_auc_precision_recall(result, self.mask).item()
            self.max_auc_seen = max(self.max_auc_seen, auc)
            return auc * 800
        return 40

    def take_action(self, action_idx: int) -> bool:
        action = self.actions[action_idx]
        if action == "(":
            self.parentheses_level += 1
        elif action == ")":
            self.parentheses_level -= 1
        self.cur_exp.append(action)

        return action == "="

    def get_valid_actions(self):
        if len(self.cur_exp) == self.max_exp_len - 1:
            return {self.actions.index("=")}
        acts_1 = []
        for i, act in enumerate(self.actions):
            if act[0] == "c" or act in ("("):
                acts_1.append(i)
        acts_2 = list(set(range(len(self.actions))) - set(acts_1))

        if not self.cur_exp:
            return acts_1

        last_act = self.cur_exp[-1]

        if last_act == "=":
            return []
        if last_act in list("(+-*/"):
            return acts_1

        if last_act == "sq" or last_act == "sqrt":
            acts_2.remove(self.actions.index("sq"))
            acts_2.remove(self.actions.index("sqrt"))
        if self.parentheses_level <= 0:
            acts_2.remove(self.actions.index(")"))

        return acts_2

    def get_invalid_actions(self):
        return list(set(range(len(self.actions))) - set(self.get_valid_actions()))

    def copy(self):
        return deepcopy(self)


def get_precision_recall(result: np.ndarray, mask: np.ndarray, threshold: float = 0.5):
    pred_mask = result > threshold
    tp = np.logical_and(pred_mask, mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), mask).sum()

    return tp / (tp + fp + 0.0001), tp / (tp + fn + 0.0001)


def get_auc_precision_recall(result: np.ndarray, mask: np.ndarray):
    tot_pr = 0
    for thresh in np.arange(0.1, 1, 0.1):
        tot_pr += get_precision_recall(result, mask, thresh)[0]

    return tot_pr / 9
