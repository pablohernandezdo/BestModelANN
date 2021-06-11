import torch.nn as nn
import torch.nn.functional as F


class ANN_6k_6k(nn.Module):
    def __init__(self):
        super(ANN_6k_6k, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 6000)
        self.l3 = nn.Linear(6000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_6k_5k(nn.Module):
    def __init__(self):
        super(ANN_6k_5k, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 5000)
        self.l3 = nn.Linear(5000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_6k_4k(nn.Module):
    def __init__(self):
        super(ANN_6k_4k, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 4000)
        self.l3 = nn.Linear(4000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_6k_3k(nn.Module):
    def __init__(self):
        super(ANN_6k_3k, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 3000)
        self.l3 = nn.Linear(3000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_6k_2k(nn.Module):
    def __init__(self):
        super(ANN_6k_2k, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 2000)
        self.l3 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_6k_1k(nn.Module):
    def __init__(self):
        super(ANN_6k_1k, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_5k_5k(nn.Module):
    def __init__(self):
        super(ANN_5k_5k, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 5000)
        self.l3 = nn.Linear(5000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_5k_4k(nn.Module):
    def __init__(self):
        super(ANN_5k_4k, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 4000)
        self.l3 = nn.Linear(4000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_5k_3k(nn.Module):
    def __init__(self):
        super(ANN_5k_3k, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 3000)
        self.l3 = nn.Linear(3000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_5k_2k(nn.Module):
    def __init__(self):
        super(ANN_5k_2k, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 2000)
        self.l3 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_5k_1k(nn.Module):
    def __init__(self):
        super(ANN_5k_1k, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_4k_4k(nn.Module):
    def __init__(self):
        super(ANN_4k_4k, self).__init__()

        self.l1 = nn.Linear(6000, 4000)
        self.l2 = nn.Linear(4000, 4000)
        self.l3 = nn.Linear(4000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_4k_3k(nn.Module):
    def __init__(self):
        super(ANN_4k_3k, self).__init__()

        self.l1 = nn.Linear(6000, 4000)
        self.l2 = nn.Linear(4000, 3000)
        self.l3 = nn.Linear(3000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_4k_2k(nn.Module):
    def __init__(self):
        super(ANN_4k_2k, self).__init__()

        self.l1 = nn.Linear(6000, 4000)
        self.l2 = nn.Linear(4000, 2000)
        self.l3 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_4k_1k(nn.Module):
    def __init__(self):
        super(ANN_4k_1k, self).__init__()

        self.l1 = nn.Linear(6000, 4000)
        self.l2 = nn.Linear(4000, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_3k_3k(nn.Module):
    def __init__(self):
        super(ANN_3k_3k, self).__init__()

        self.l1 = nn.Linear(6000, 3000)
        self.l2 = nn.Linear(3000, 3000)
        self.l3 = nn.Linear(3000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_3k_2k(nn.Module):
    def __init__(self):
        super(ANN_3k_2k, self).__init__()

        self.l1 = nn.Linear(6000, 3000)
        self.l2 = nn.Linear(3000, 2000)
        self.l3 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_3k_1k(nn.Module):
    def __init__(self):
        super(ANN_3k_1k, self).__init__()

        self.l1 = nn.Linear(6000, 3000)
        self.l2 = nn.Linear(3000, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_2k_2k(nn.Module):
    def __init__(self):
        super(ANN_2k_2k, self).__init__()

        self.l1 = nn.Linear(6000, 2000)
        self.l2 = nn.Linear(2000, 2000)
        self.l3 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_2k_1k(nn.Module):
    def __init__(self):
        super(ANN_2k_1k, self).__init__()

        self.l1 = nn.Linear(6000, 2000)
        self.l2 = nn.Linear(2000, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class ANN_1k_1k(nn.Module):
    def __init__(self):
        super(ANN_1k_1k, self).__init__()

        self.l1 = nn.Linear(6000, 1000)
        self.l2 = nn.Linear(1000, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)

