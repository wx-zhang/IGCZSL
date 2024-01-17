import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, feature_size=2048, att_size=85):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(2 * att_size, 1024)
        self.fc2 = nn.Linear(1024, feature_size)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, noise, att):
        if len(att.shape) == 3:
            h = torch.cat((noise, att), 2)
        else:
            h = torch.cat((noise, att), 1)
        feature = torch.relu(self.fc1(h))
        feature = torch.sigmoid(self.fc2(feature))
        return feature


class Discriminator(nn.Module):
    def __init__(self, feature_size=2048, att_size=85):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(att_size, 1024)
        self.fc2 = nn.Linear(1024, feature_size)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, att):
        att_embed = torch.relu(self.fc1(att))
        att_embed = torch.relu(self.fc2(att_embed))
        return att_embed