import torch.nn as nn

class IndexRLAgent(nn.Module):
    def __init__(self, action_size: int, state_size: int):
        super(IndexRLAgent, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
            nn.Softmax(dim=0)
        )
    
    def forward(self, x):
        return self.backbone(x)
