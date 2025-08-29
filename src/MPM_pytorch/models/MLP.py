
import torch.nn.functional as F
import torch.nn as nn



class MLP(nn.Module):

    def __init__(self, input_size=None, output_size=None, nlayers=None, hidden_size=None, device=None, activation=None, initialisation=None):

        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size, device=device))
        if nlayers > 2:
            for i in range(1, nlayers - 1):
                layer = nn.Linear(hidden_size, hidden_size, device=device)
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)
                self.layers.append(layer)
        layer = nn.Linear(hidden_size, output_size, device=device)

        if initialisation == 'zeros':
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif initialisation == 'ones':
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        else :
            nn.init.normal_(layer.weight, std=0.1)
            nn.init.zeros_(layer.bias)

        self.layers.append(layer)

        if activation=='none':
            self.activation = lambda x: x
        elif activation=='tanh':
            self.activation = F.tanh
        else:
            self.activation = F.relu

    def forward(self, x):
        for l in range(len(self.layers) - 1):
            x = self.layers[l](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


if __name__ == '__main__':


    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from tqdm import trange
    import matplotlib
    import matplotlib.pyplot as plt
    import torch

    device='cuda:0'


    matplotlib.use("Qt5Agg")

    nlayers=10

    model = MLP(input_size=1, output_size=1, nlayers=nlayers, hidden_size=64, device='cuda:0')
    optimizer = optim.Adam(model.parameters(), lr=1E-4)
    model.train()

    max_radius=0.075

    r = torch.linspace(0, max_radius, 1000, dtype=torch.float32, device=device)

    p = torch.tensor([0.03, 0.03, 100, 1.0], device='cuda:0')
    y = p[0] * torch.tanh((r - p[1]) * p[2])
    y = y[:,None]

    # fig = plt.figure()
    # p = torch.tensor([0.03, 0.03, 100, 1.0], device='cuda:0')
    # out = p[0] * torch.tanh((r-p[1])*p[2])
    # plt.plot(r.detach().cpu().numpy(), out.detach().cpu().numpy(), linewidth=2)
    # plt.tight_layout()

    for epoch in trange(1000):

        optimizer.zero_grad()

        pred = model(r[:, None])

        loss = (pred-y).norm(2)

        loss.backward()
        optimizer.step()

    fig = plt.figure()
    plt.plot(r.detach().cpu().numpy(), pred.detach().cpu().numpy(), linewidth=2)
    plt.plot(r.detach().cpu().numpy(), y.detach().cpu().numpy(), linewidth=2)
    plt.tight_layout()





    model = MLP(input_size=1, output_size=1, nlayers=nlayers, hidden_size=64, device='cuda:0')
    optimizer = optim.Adam(model.parameters(), lr=1E-4)
    model.train()

    max_radius=0.075
    sigma = 0.005

    r = torch.linspace(0, max_radius, 1000, dtype=torch.float32, device=device)

    p = torch.tensor([1.6233, 1.0413, 1.6012, 1.5615], device=device)
    y = r * (p[0] * torch.exp(-r ** (2 * p[1]) / (2 * sigma ** 2)) - p[2] * torch.exp(-r ** (2 * p[3]) / (2 * sigma ** 2)))
    y = y[:,None]

    # fig = plt.figure()
    # p = torch.tensor([0.03, 0.03, 100, 1.0], device='cuda:0')
    # out = p[0] * torch.tanh((r-p[1])*p[2])
    # plt.plot(r.detach().cpu().numpy(), out.detach().cpu().numpy(), linewidth=2)
    # plt.tight_layout()

    for epoch in trange(1000):

        optimizer.zero_grad()

        pred = model(r[:, None])

        loss = (pred-y).norm(2)

        loss.backward()
        optimizer.step()

    fig = plt.figure()
    plt.plot(r.detach().cpu().numpy(), pred.detach().cpu().numpy(), linewidth=2)
    plt.plot(r.detach().cpu().numpy(), y.detach().cpu().numpy(), linewidth=2)
    plt.tight_layout()













