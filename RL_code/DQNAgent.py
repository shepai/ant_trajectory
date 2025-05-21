
import random
import numpy as np
from collections import deque
import torch.optim as optim
from RL_code.AntAgentCNN import AntAgentCNN
import torch
import torch.nn.functional as F
''' 
Okay so, this class is for the deep q network agent and its behaviours (action selection, experience storage, and learning)


'''

class DQNAgent:
    def __init__(self, state_shape, num_actions, device):
        self.device = device
        self.num_actions = num_actions
      # policy network is used to select actions based on the current state
        self.policy_net = AntAgentCNN(input_channels=state_shape[0], num_actions=num_actions).to(device)
        # separate network that computes the target q values for stability, and is updated with the weights of the policy network
        self.target_net = AntAgentCNN(input_channels=state_shape[0], num_actions=num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4) #here we use the Adam optimiser to update the policy net's weights
        #memory stores past experiences for replay
        self.memory = deque(maxlen=10000)
        # here we have the hyperparameters we can play with
        self.batch_size = 64 # experiences sampled per training step
        self.gamma = 0.99 # discount factor for future rewards
        self.epsilon = 1.0 # exploration rate for EPSILON GREEDY POLICY my fav
        self.epsilon_decay = 0.995 
        self.epsilon_min = 0.1 
        self.update_target_every = 10 # num of steps between target network updates
        self.steps_done = 0

  # this is where the epsilon-greedy policy is implemented, with probability epsilon, a random action is selected
  #(this is for the whole exploration-exploitation thing happens, by adding some randomness the model can do better) 
    def step(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item() #the action with the highest Q value is selected 
          
# here we store a tuple of state, action, reward, next_state, done, in the replay memory for when it samples during training
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
      #check if there are enough experiences in the replay memory to form a batch 
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size) # randomly samples a batch of experiences from hte replay memories
        states, actions, rewards, next_states, dones = zip(*batch)
        #data prep, the sampled experiences are unpacked in the different variables
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        #actions=F.one_hot(actions, num_classes=3).float()
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # compute current q vals using the policy net
        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        #loss computation & backprop -- Mean Squared Error (MSE) loss between the current Q-values and the target Q-values is computed
        loss = F.mse_loss(current_q, target_q) 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


      #the target network's weights are updated to match the policy network's weights. 
      #(This decouples the target from the rapidly fluctuating policy network, aiding in stable learning.)
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        #epsilon decays
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class ContinuousAgent:
    def __init__(self, state_shape, device):
        self.device = device
        self.policy_net = AntAgentCNN(input_channels=state_shape[0], num_actions=2).to(device)  # 2 for [vx, vy]
        self.target_net = AntAgentCNN(input_channels=state_shape[0], num_actions=2).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = deque(maxlen=10000)

        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.update_target_every = 10
        self.steps_done = 0

    def step(self, state):
        if random.random() < self.epsilon:
            return np.random.uniform(-0.5, 0.5, size=2)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.policy_net(state)
            return action.cpu().numpy().flatten()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Predict current actions (this is like an actor outputting actions)
        current_actions = self.policy_net(states)
        # Predict next actions from target network (could be used in target estimation)
        next_actions = self.target_net(next_states).detach()

        # Calculate target Q values (mock version assuming reward + future prediction)
        target_actions = rewards + self.gamma * (1 - dones) * next_actions.norm(dim=1, keepdim=True)

        # Loss: how far the predicted actions are from target actions (mock)
        predicted_action_values = current_actions.norm(dim=1, keepdim=True)
        loss = F.mse_loss(predicted_action_values, target_actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay