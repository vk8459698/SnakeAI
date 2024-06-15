import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        os.makedirs(model_folder_path, exist_ok=True)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))


class QTrainer:
    def __init__(self, model: Linear_QNet, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state_old, action, reward, state_new, done):
        state_old = torch.tensor(state_old, dtype=torch.float)
        state_new = torch.tensor(state_new, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state_old.shape) == 1:
            # received only one state
            # need to change to shape (1, x)
            # appends one dimension at beginning of each tensor
            state_old = torch.unsqueeze(state_old, 0)
            state_new = torch.unsqueeze(state_new, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            # tuple with one dimension
            done = (done, )

        # 1: predict Q values with current state
        pred_action: torch.Tensor = self.model(state_old)  # list

        # 2: Q_new = r + y * max(next_predicted_Q_value) -> only do this if not done
        target = pred_action.clone()
        for index in range(len(done)):
            Q_new = reward[index]
            if not done[index]:
                Q_new = reward[index] + self.gamma * \
                    torch.max(self.model(state_new))

            target[index][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss: torch.Tensor = self.criterion(target, pred_action)
        loss.backward()
        self.optimizer.step()
