from torch import nn
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self, d_in, d_h, d_out, dp):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.fc2 = nn.Linear(d_h, d_out)
        self.dp = nn.Dropout(dp)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp(x)

        x = self.fc2(x)

        return x
