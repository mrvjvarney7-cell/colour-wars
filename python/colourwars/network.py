"""AlphaZero-style network for Colour Wars.

Input: (NUM_PLANES, rows, cols) board encoding, always from the current
mover's perspective (see env.encode_state).
Output:
  - policy: logits over rows*cols flat actions.
  - value: MAX_PLAYERS-length vector of predicted outcomes in [-1, 1],
    interpreted RELATIVE to the current mover (index 0 = current mover,
    index k = the player k slots after them in id-rotation order, see
    env.encode_state / env.action_to_relative_owner_perm). This is what
    makes a single network usable across 2/3/4-player games without a
    separate head per player count.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from colourwars.env import NUM_PLANES, MAX_PLAYERS
from colourwars.game import ROWS, COLS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ColourWarsNet(nn.Module):
    def __init__(
        self,
        rows: int = ROWS,
        cols: int = COLS,
        channels: int = 64,
        num_res_blocks: int = 6,
    ):
        super().__init__()
        self.rows = rows
        self.cols = cols

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * rows * cols, rows * cols)

        # Value head (per-player, relative-to-mover)
        self.value_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * rows * cols, 128)
        self.value_fc2 = nn.Linear(128, MAX_PLAYERS)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.res_blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    @torch.no_grad()
    def predict(self, state_tensor: torch.Tensor, device: torch.device):
        """Single-state convenience inference. state_tensor: (NUM_PLANES, R, C)
        on CPU. Returns (policy_logits[R*C] cpu numpy, value[MAX_PLAYERS] cpu numpy)."""
        self.eval()
        x = state_tensor.unsqueeze(0).to(device)
        policy_logits, value = self.forward(x)
        return policy_logits[0].cpu().numpy(), value[0].cpu().numpy()
