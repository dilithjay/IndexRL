import numpy as np
import torch

from environment import IndexRLEnv

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/home/dilith/Projects/IndexRL/models/models-pt-10c-of/model_23_loss-0.33211189108590283.pt"

max_exp_len = 16
image = np.ones((10, 2, 2))
mask = np.ones((2, 2))

n_channels = image.shape[0]
action_list = list("()+-*/=") + ["sq", "sqrt"] + [f"c{c}" for c in range(n_channels)]
env = IndexRLEnv(action_list, max_exp_len, False, masked_actions=["sq", "sqrt"])
cur_state = env.reset(image, mask)

agent = torch.load(model_path)
for act in env.get_valid_actions():
    env = IndexRLEnv(action_list, max_exp_len, False, masked_actions=["sq", "sqrt"])
    cur_state = env.reset(image, mask)
    cur_state, _, _ = env.step(act)
    while True:
        state = torch.tensor(np.array([cur_state])).int().to(device)
        with torch.no_grad():
            probs = agent.generate_single(state).squeeze()

        invalid_acts = env.get_invalid_actions()
        probs[invalid_acts] = -1
        # print(tuple(zip(env.actions, probs.tolist())))

        action_idx = probs.argmax()

        cur_state, reward, done = env.step(action_idx)

        if done:
            print(env.state_to_expression(cur_state))
            print()
            break
