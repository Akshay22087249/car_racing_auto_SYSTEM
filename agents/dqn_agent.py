"""Deep Q-Network (DQN) agent for CarRacing-v3.

Implements DQN from scratch using only PyTorch and NumPy.
No Stable Baselines or other RL libraries are used.

Architecture:
    - Convolutional Neural Network (CNN) to process 96x96 pixel frames
    - Experience replay buffer to break correlation between samples
    - Target network to stabilise training
    - Epsilon-greedy exploration strategy

Reference:
    Mnih et al. (2015). Human-level control through deep reinforcement
    learning. Nature, 518(7540), 529-533.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Action space ──────────────────────────────────────────────────────────────
DO_NOTHING = 0
LEFT       = 1
RIGHT      = 2
GAS        = 3
BRAKE      = 4
NUM_ACTIONS = 5


class DQNNetwork(nn.Module):
    """CNN that maps a stack of greyscale frames to Q-values.

    Input:  (batch, frames, 84, 84)  — preprocessed greyscale frames
    Output: (batch, NUM_ACTIONS)     — estimated Q-value per action
    """

    def __init__(self, n_frames: int = 4, n_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # Conv1: 84x84 -> 20x20
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Conv2: 20x20 -> 9x9
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Conv3: 9x9 -> 7x7
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise pixel values to [0, 1]
        x = x.float() / 255.0
        return self.fc(self.conv(x))


class ReplayBuffer:
    """Fixed-size circular buffer that stores experience tuples.

    Each entry is (state, action, reward, next_state, done).
    Sampling a random minibatch breaks the temporal correlation
    between consecutive transitions, which stabilises training.
    """

    def __init__(self, capacity: int = 50_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.uint8),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.uint8),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN agent with experience replay and a target network.

    Hyperparameters
    ---------------
    lr              : learning rate for Adam optimiser
    gamma           : discount factor for future rewards
    epsilon_start   : initial exploration rate
    epsilon_end     : minimum exploration rate
    epsilon_decay   : multiplicative decay per step
    batch_size      : number of transitions per gradient update
    target_update   : how often (in steps) to copy online -> target network
    buffer_capacity : maximum number of transitions in replay buffer
    n_frames        : number of stacked frames fed to the network
    """

    def __init__(
        self,
        lr: float            = 1e-4,
        gamma: float         = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float   = 0.05,
        epsilon_decay: float = 0.9995,
        batch_size: int      = 64,
        target_update: int   = 1000,
        buffer_capacity: int = 50_000,
        n_frames: int        = 4,
        device: str          = "auto",
    ) -> None:

        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.n_frames      = n_frames
        self.steps_done    = 0

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"  [DQN] Using device: {self.device}")

        # Networks
        self.online_net = DQNNetwork(n_frames).to(self.device)
        self.target_net = DQNNetwork(n_frames).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimiser
        self.optimiser = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        # Frame stack (ring buffer)
        self._frame_stack: deque = deque(maxlen=n_frames)

    # ── Preprocessing ─────────────────────────────────────────────────────────

    @staticmethod
    def preprocess(frame: np.ndarray) -> np.ndarray:
        """Convert 96x96x3 RGB frame to 84x84 greyscale uint8."""
        import cv2
        grey = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(grey, (84, 84), interpolation=cv2.INTER_AREA)
        return resized  # shape (84, 84), dtype uint8

    def reset_frame_stack(self, frame: np.ndarray) -> np.ndarray:
        """Fill the frame stack with copies of the first frame."""
        processed = self.preprocess(frame)
        for _ in range(self.n_frames):
            self._frame_stack.append(processed)
        return np.array(self._frame_stack)  # (n_frames, 84, 84)

    def push_frame(self, frame: np.ndarray) -> np.ndarray:
        """Add a new frame and return the updated stack."""
        self._frame_stack.append(self.preprocess(frame))
        return np.array(self._frame_stack)  # (n_frames, 84, 84)

    # ── Action selection ──────────────────────────────────────────────────────

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection.

        During training a random action is chosen with probability
        epsilon (exploration). Otherwise the network's greedy action
        is used (exploitation).
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)

        with torch.no_grad():
            tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(tensor)
            return int(q_values.argmax(dim=1).item())

    # ── Learning ──────────────────────────────────────────────────────────────

    def learn(self) -> float | None:
        """Sample a minibatch and perform one gradient update.

        Returns the loss value, or None if the buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states      = torch.from_numpy(states).to(self.device)
        actions     = torch.from_numpy(actions).to(self.device)
        rewards     = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones       = torch.from_numpy(dones).to(self.device)

        # Current Q-values for the taken actions
        q_current = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values using the Bellman equation
        with torch.no_grad():
            q_next   = self.target_net(next_states).max(dim=1).values
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = self.loss_fn(q_current, q_target)

        self.optimiser.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimiser.step()

        # Decay epsilon after each learning step
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps_done += 1

        # Periodically copy online network weights to target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push a transition into the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save(
            {
                "online_net":  self.online_net.state_dict(),
                "target_net":  self.target_net.state_dict(),
                "optimiser":   self.optimiser.state_dict(),
                "epsilon":     self.epsilon,
                "steps_done":  self.steps_done,
            },
            path,
        )
        print(f"  [DQN] Saved checkpoint to {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimiser.load_state_dict(checkpoint["optimiser"])
        self.epsilon    = checkpoint["epsilon"]
        self.steps_done = checkpoint["steps_done"]
        print(f"  [DQN] Loaded checkpoint from {path} (step {self.steps_done})")
