# SIREN network
# Code adapted from the following GitHub repository:
# https://github.com/vsitzmann/siren?tab=readme-ov-file
import os

import numpy as np
import torch
import torch.nn as nn

# from ParticleGraph.generators.utils import get_time_series
import matplotlib
from matplotlib import pyplot as plt
from tifffile import imread, imwrite as imsave
from tqdm import trange
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# import skimage
# from torchvision.transforms import Resize, Compose, ToTensor, Normalize



class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output


class small_Siren(nn.Module):
    def __init__(self, in_features=1, hidden_features=128, hidden_layers=3, out_features=1, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()

        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        layers.append(nn.Linear(hidden_features, out_features))  # final linear layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Siren_Network(nn.Module):
    def __init__(self, image_width, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., device='cuda:0'):
        super().__init__()

        self.device = device 
        self.image_width = image_width

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)

        self.net = self.net.to(device)

        self.coords = None

    @property
    def values(self):
        # Call forward method
        output, coords = self.__call__()
        return output.squeeze().reshape(self.image_width, self.image_width)
    
    def coordinate_grid(self, n_points):
        coords = np.linspace(0, 1, n_points, endpoint=False)
        xy_grid = np.stack(np.meshgrid(coords, coords), -1)
        xy_grid = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(self.device)
        return xy_grid
    
    def get_mgrid(self, sidelen, dim=2, enlarge=False):
        '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
        sidelen: int
        dim: int'''
        if enlarge:
            # tensors = tuple(dim * [torch.linspace(-0.2, 1.2, steps=sidelen*20)])
            tensors = tuple(dim * [torch.linspace(0, 1, steps=sidelen*20)])
        else:
            tensors = tuple(dim * [torch.linspace(0, 1, steps=sidelen)])

        mgrid = torch.stack(torch.meshgrid(*tensors,indexing="ij"), dim=-1)
        mgrid = mgrid.reshape(-1, dim)
        return mgrid

    def forward(self, coords=None, time=None, enlarge=False):

        if coords is None:
            coords = self.get_mgrid(self.image_width, dim=2, enlarge=enlarge).to(self.device)
            if time != None:
               coords = torch.cat((coords, time * torch.ones_like(coords[:, 0:1])), 1)

        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output

        output = self.net(coords)
        return output


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img


if __name__ == '__main__':



    device = 'cuda:0'
    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    model = Siren(in_features = 1, hidden_features = 256 , hidden_layers = 3, out_features = 300, outermost_linear=True, first_omega_0=30., hidden_omega_0=30.)
    model = model.to(device=device)
    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())

    x_list = np.load('/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/CElegans/CElegans_c1/x_list_0.npy')
    x_list = np.nan_to_num(x_list, nan=0.0)

    activity = torch.tensor(x_list,device=device)
    activity = activity[:, :, 6:7].squeeze()
    activity = activity.t()
    activity = torch.nan_to_num(activity, nan=0.0)
    y = activity

    # plt.imshow(activity.detach().cpu().numpy(), cmap='grey')
    # plt.show()

    n_frames = 958
    batchsize = 100

    for epoch in trange(1000000//batchsize):
        optimizer.zero_grad()

        loss = 0
        for batch in range(batchsize):

            k = np.random.randint(n_frames - 1)
            x = torch.tensor(x_list[k], dtype=torch.float32, device=device)

            t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
            missing_activity = model(t) ** 2

            loss = loss + (missing_activity - x[:, 6].clone().detach()).norm(2)

        loss.backward()
        optimizer.step()

        if epoch % (10000//batchsize) == 0:
            print(f"Epoch {epoch}, Loss {loss.item()}")
            with torch.no_grad():
                t = torch.linspace(0, 1, n_frames, dtype=torch.float32, device=device).unsqueeze(1)
                pred = model(t) ** 2
                pred = pred.t()
            fig = plt.figure(figsize=(16, 8))
            plt.imshow(pred.detach().cpu().numpy(), cmap='grey')
            plt.show()

    #####################################

    model_siren = Siren(in_features = 1, hidden_features = 256 , hidden_layers = 3, out_features = 287400, outermost_linear=True, first_omega_0=30., hidden_omega_0=30.)
    model_siren = model_siren.to(device=device)
    optimizer = torch.optim.Adam(lr=1e-4, params=model_siren.parameters())

    x_list = np.load('/groups/saalfeld/home/allierc/Py/ParticleGraph/graphs_data/CElegans/CElegans_c1/x_list_0.npy')
    activity = torch.tensor(x_list,device=device)
    activity = activity[:, :, 6:7].squeeze()
    activity = activity.t()
    activity = torch.nan_to_num(activity, nan=0.0)
    y = activity.flatten()


    for epoch in trange(10000):
        optimizer.zero_grad()

        t = torch.tensor([1],dtype=torch.float32, device=device)
        x = model_siren(t) ** 2

        loss = (x - y).norm(2)

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss {loss.item()}")
            pred = model_siren(t) ** 2
            # pred = torch.reshape(pred, (256, 256))
            pred = torch.reshape(pred, (300, 958))
            fig = plt.figure(figsize=(16, 8))
            plt.imshow(pred.detach().cpu().numpy(), cmap='grey')
            plt.show()

    #####################################

    model_siren = Siren_Network(image_width=256, in_features=2, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True)
    model_siren = model_siren.to(device=device)
    optimizer = torch.optim.Adam(lr=1e-4, params=model_siren.parameters())

    i0 = imread('data/pics_boat.tif')

    y = torch.tensor(i0, dtype=torch.float32, device=device)
    y = y.flatten()
    y = y[:,None]

    coords = get_mgrid(256, dim=2)
    coords = coords.to('cuda:0')

    print(coords.device, y.device)


    for epoch in trange(10000):
        optimizer.zero_grad()

        x = model_siren()**2

        loss = (x - y).norm(2)

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss {loss.item()}")
            pred = model_siren()**2
            pred = torch.reshape(pred, (256, 256))
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(pred.detach().cpu().numpy(), cmap='grey')
            plt.show()

            # plt.scatter(y.detach().cpu().numpy(),x.detach().cpu().numpy(),c='k',s=1)
            # plt.savefig(f"tmp/output_{epoch}.png")


    #####################################


    cameraman = ImageFitting(256)
    dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

    img_siren = Siren(in_features=2, out_features=1, hidden_features=256,
                      hidden_layers=3, outermost_linear=True)
    img_siren.cuda()

    total_steps = 500  # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 10

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    for step in range(total_steps):
        model_output = img_siren(model_input)
        loss = ((model_output - ground_truth) ** 2).mean()

        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            # img_grad = gradient(model_output, coords)
            # img_laplacian = laplace(model_output, coords)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(model_output.cpu().view(256, 256).detach().numpy())
            # axes[1].imshow(img_grad.norm(dim=-1).cpu().view(256, 256).detach().numpy())
            # axes[2].imshow(img_laplacian.cpu().view(256, 256).detach().numpy())
            plt.show()
            plt.savefig(f"tmp/output_{step}.png")
            plt.close()

        optim.zero_grad()
        loss.backward()
        optim.step()
