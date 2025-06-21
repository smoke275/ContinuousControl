import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from ReplayBuffer import ReplayBuffer
from OUNoise import OUNoise
from model import Actor, Critic

# Hyperparameters
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1.5e-4       # learning rate of the actor 
LR_CRITIC = 1.5e-4      # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """DDPG Agent that learns to control continuous actions."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize the DDPG Agent.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Networks (local and target)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Networks (local and target)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), 
                                          lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process for exploration
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience and learn from a batch of experiences."""
        # Store experience in replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # Learn from experiences if we have enough samples
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=False):
        """Return action for given state using current policy."""
        state = torch.from_numpy(state).float().to(device)
        
        # Set actor to evaluation mode
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()  # Back to training mode
        
        # Add noise for exploration during training
        if add_noise:
            action += self.noise.sample()
        
        # Clip actions to valid range [-1, 1]
        return np.clip(action, -1, 1)

    def reset(self):
        """Reset the noise process (call at start of each episode)."""
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update actor and critic networks using batch of experiences.
        
        DDPG Algorithm:
        1. Update Critic: minimize TD error using target networks
        2. Update Actor: maximize Q-value of actions from current policy
        3. Soft update target networks
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- Update Critic ---------------------------- #
        # Get next actions from target actor
        actions_next = self.actor_target(next_states)
        
        # Get Q-values for next states and actions from target critic
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Calculate target Q-values (Bellman equation)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Get current Q-values from local critic
        Q_expected = self.critic_local(states, actions)
        
        # Calculate critic loss (TD error)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- Update Actor ---------------------------- #
        # Get actions from current policy
        actions_pred = self.actor_local(states)
        
        # Calculate actor loss (negative because we want to maximize Q-value)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- Update Target Networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Slowly update target network parameters.
        
        Instead of copying weights directly, we blend:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        This makes training more stable.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)