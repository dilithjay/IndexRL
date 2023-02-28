import gym
from copy import deepcopy
import numpy as np

from expression_handler import check_unitless_validity, eval_expression


class IndexRLEnv(gym.Env):
    def __init__(
        self,
        discrete_actions: list,
        max_exp_len: int = 100,
        ohe: bool = True,
        masked_actions: list = None,
        unitless: bool = False,
    ):
        super(IndexRLEnv, self).__init__()
        self.actions = discrete_actions
        self.image = None
        self.mask = None
        self.cur_exp = []
        self.parentheses_level = 0
        self.max_exp_len = max_exp_len
        self.ohe = ohe
        self.masked_actions = masked_actions
        self.unitless = unitless

        self.best_reward = 0
        self.best_exp = []

    def get_cur_state(self):
        if self.ohe:
            cur_exp_indices = [self.actions.index(act) for act in self.cur_exp] + [0] * (
                self.max_exp_len - len(self.cur_exp)
            )
            enc_state = np.zeros((self.max_exp_len, len(self.actions)))
            enc_state[np.arange(self.max_exp_len), cur_exp_indices] = 1
            return enc_state.flatten()
        else:
            cur_exp_indices = [self.actions.index(act) for act in self.cur_exp]
            return np.array(cur_exp_indices)

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
        result = False
        if not self.unitless or (self.unitless and check_unitless_validity(self.cur_exp)):
            result = eval_expression(self.cur_exp, self.image.squeeze())
        if result is False and done:
            return -1
        if done:
            if len(self.mask) > 2:
                reward = get_auc_precision_recall(result, self.mask > 0)
                # reward = (np.abs(result - self.mask) < 0.1).sum() / len(self.mask)
                self.best_reward = max(self.best_reward, reward)
                self.best_exp = self.cur_exp
                return reward
            else:  # Pretraining stage
                if len(self.cur_exp) < 7:
                    return -1
                return 0.4 * len(self.cur_exp)
        return 0

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

        # Include all the channels and opening brackets in action set 1
        acts_1 = []
        for i, act in enumerate(self.actions):
            if act[0] == "c" or act in ("("):
                acts_1.append(i)

        # Remove actions specified as masked actions from acts_1
        if self.masked_actions:
            for act in self.masked_actions:
                idx = self.actions.index(act)
                if idx in acts_1:
                    acts_1.remove(idx)

        # Include the inverse of the action set 1 in action set 2
        acts_2 = list(set(range(len(self.actions))) - set(acts_1))

        # Allow action set 1 when just starting the episode
        if not self.cur_exp:
            return acts_1

        last_act = self.cur_exp[-1]

        # Disallow any other actions if episode ended with "="
        if last_act == "=":
            return []

        # If last action was one of the following, allow selecting channels or an open parenthesis
        if last_act in list("(+-*/"):
            return acts_1

        # Disallow consecutive squares, square roots, or combinations of them.
        if last_act == "sq" or last_act == "sqrt":
            acts_2.remove(self.actions.index("sq"))
            acts_2.remove(self.actions.index("sqrt"))

        # Remove actions specified as masked actions from acts_2
        if self.masked_actions:
            for act in self.masked_actions:
                idx = self.actions.index(act)
                if idx in acts_2:
                    acts_2.remove(idx)

        # Disallow closing the brackets after just one character
        if len(self.cur_exp) > 1 and self.cur_exp[-2] == "(":
            acts_2.remove(self.actions.index(")"))

        # Disallow closing paranthesis if there are no open paranthesis
        if self.parentheses_level <= 0:
            acts_2.remove(self.actions.index(")"))
        else:
            acts_2.remove(self.actions.index("="))

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
    tot_score = 0
    for thresh in np.arange(-1, 1, 0.1):
        pr, rec = get_precision_recall(result, mask, thresh)
        tot_score += min(pr, rec)

    return tot_score / 20
