import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, action_dim):
        super().__init__()
        self.channels, self.height, self.width = input_shape

        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Calculate flattened size for the fully connected layer
        def conv_output_size(h, w):
            x = torch.randn(1, self.channels, h, w) # dummy tensor
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            return x.flatten().shape[0] # size (64*6*7 = 2688)
    
        self.fc_input_dim = conv_output_size(self.height, self.width)

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.fc_out = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten the output of conv layers
        x = F.relu(self.fc1(x))
        return self.fc_out(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # Unzip and convert to numpy arrays
        states, actions, rewards, next_states, dones = map(np.array, zip(*transitions))

        # Add channel dimension for CNN: (B, H, W) -> (B, 1, H, W)
        states = np.expand_dims(states, axis=1)
        next_states = np.expand_dims(next_states, axis=1)

        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, board_shape, n_actions, device,
                learning_rate=1e-4,
                gamma=0.99,
                replay_buffer_capacity=100000,
                batch_size=64,
                target_update_freq=1000):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_step_counter = 0

        # For CNN, input_shape is (channels, height, width)
        # 1 channel (player 1 = 1, player -1 = -1, empty = 0)
        cnn_input_shape = (1, board_shape[0], board_shape[1])

        self.online_net = DQN(cnn_input_shape, n_actions).to(device) # student
        self.target_net = DQN(cnn_input_shape, n_actions).to(device) # teacher
        self.update_target()
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
    
    def update_target(self):
        """Copy all weights and biases from student to the teacher"""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def select_action(self, state_2d, epsilon, valid_actions):
        """
        Choose and action with ε-greedy policy, filtering out illegal moves.
        :param state_2d: np.array state (rows, cols)
        :param epsilon: exploration probability
        :param valid_actions: list of legal action indices
        """
        if random.random() < epsilon:
            return random.choice(valid_actions)
        else:
            self.online_net.eval() # Eval mode for inference ("thinking-only")
            with torch.no_grad(): # Do not calculate gradients for the following operations
                state_tensor = torch.tensor(state_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0) .to(self.device)
                q_values = self.online_net(state_tensor).cpu().squeeze(0)

                # Mask invalid actions by setting their Q-values to -infinity
                masked_q_values = torch.full_like(q_values, -float('-inf'))
                # Copy Q-values for valid actions
                valid_actions_tensor = torch.tensor(valid_actions, dtype=torch.long)
                masked_q_values[valid_actions_tensor] = q_values[valid_actions_tensor]

                best_action = int(masked_q_values.argmax().item())
            
            self.online_net.train() # Back to training mode
            return best_action
    

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None # Not enough sample to learn
        
        self.online_net.train() 

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32).to(self.device) # (B, 1, H, W)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device) # (B, 1, H, W)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device) # (B, 1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device) # (B, 1)
        dones_t = torch.tensor(dones, dtype=torch.bool).unsqueeze(1).to(self.device) # (B, 1)

        # --- Double DQN & Self-Play Target Calculation ---
        with torch.no_grad():
            # 1. Select best actions in next_states_t using the online_net
            next_q_values_online = self.online_net(next_states_t)
            best_next_actions = next_q_values_online.argmax(dim=1, keepdim=True)

            # 2. Evaluate these best_next_actions using the target_net
            next_q_values_target = self.target_net(next_states_t).gather(1, best_next_actions)

            # 3. Adjust for self-play: what's good for opponent is bad for current player
            adjusted_next_q = -next_q_values_target

            # 4. Compute the target Q-value
            # If done, target_q is just the reward.
            # Otherwise, it's reward + gamma * (adjusted_next_q)
            target_q_values = rewards_t + (~dones_t) * self.gamma * adjusted_next_q
        
        # --- Current Q values ---
        # Q-values for the actions actually taken, from online_net
        current_q_values = self.online_net(states_t).gather(1, actions_t)

        # --- Compute loss ---
        loss = F.smooth_l1_loss(current_q_values, target_q_values) # Huber loss, often more stable

        # --- Optimize the model ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0) 
        self.optimizer.step()

        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.update_target()
        
        return loss.item()
    

    def store_transitions(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

