import numpy as np
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process for adding noise to actions in DDPG."""
    
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)  # Mean value
        self.theta = theta  # How fast it returns to mean
        self.sigma = sigma  # Noise intensity
        self.seed = np.random.seed(seed)
        self.reset()
    
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
    
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        # OU process: pull toward mean + add random noise
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state