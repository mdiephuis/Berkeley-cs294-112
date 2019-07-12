import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


# Model definition
class BehaviorClone(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(BehaviorClone, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.fc1 = nn.Linear(input_shape, input_shape // 2)
        self.fc2 = nn.Linear(input_shape // 2, output_shape)
        self.do = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.do(x)
        x = self.fc2(x)
        return x


def train_valid(model, loader, optimizer, loss_fn, train):

    # train / valid loop
    model.train() if train else model.eval()
    batch_loss = 0
    for batch_idx, (x, y) in enumerate(loader):
        batch_size = x.size(0)
        loss = 0
        if train:
            optimizer.zero_grad()

        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        batch_loss += loss.item() / batch_size

        if train:
            loss.backward()
            optimizer.step()

    batch_loss /= (batch_idx + 1)
    return batch_loss


def save_checkpoint(state, filename):
    torch.save(state, filename)


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for sub_mod in m:
                init_weights(sub_mod)
