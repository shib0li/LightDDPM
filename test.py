import numpy as np
import torch
import torch.nn as nn
import random
import os

from sklearn.datasets import make_s_curve
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

from models.ddpm.mlp import MLPDiffusion
from infras.misc import cprint, create_path


device = torch.device('cpu')

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class DDPM(nn.Module):

    def __init__(
        self,
        s_dim,
        diff_steps,
        beta_min=1e-5,
        beta_max=5e-3,
        hidden_dim=128,
        hidden_layers=3,
    ):
        super().__init__()

        self.s_dim = s_dim
        self.diff_steps = diff_steps

        self.noise_net = MLPDiffusion(
            s_dim=self.s_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            n_steps=self.diff_steps,
        )

        betas = torch.linspace(-10,10,self.diff_steps)
        betas = torch.sigmoid(betas)*(beta_max-beta_min)+beta_min
        alphas = 1.0 - betas

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', torch.cumprod(alphas,0))
        self.register_buffer('alphas_bar_sqrt', torch.sqrt(torch.cumprod(alphas,0)))
        self.register_buffer('one_minus_alphas_bar_sqrt', torch.sqrt(1.0-torch.cumprod(alphas,0)))

        self.register_buffer('dummy', torch.tensor([]))

    def forward_diffusion(self, x0, t):
        epsilon = torch.randn_like(x0).to(self.dummy.device)
        c_mean = self.alphas_bar_sqrt[t]
        c_std = self.one_minus_alphas_bar_sqrt[t]
        xt = x0*c_mean + epsilon*c_std   # reparam trick
        return xt

    def eval_noise_loss(self, x0):

        batch_size = x0.shape[0]

        t = torch.randint(0,self.diff_steps,size=(batch_size//2,))
        t = torch.cat([t,self.diff_steps-1-t],dim=0)

        coeff_x0 = self.alphas_bar_sqrt[t].reshape([-1,1])
        coeff_epsi = self.one_minus_alphas_bar_sqrt[t].reshape([-1,1])

        epsi = torch.randn_like(x0).to(self.dummy.device)
        x = x0*coeff_x0 + coeff_epsi*epsi

        pred = self.noise_net(x, t)
        err_noise = (epsi - pred).square().mean()

        return err_noise


    def _reverse_one_step(self, xt, t):
        # reverse one step to x_{t-1}
        assert type(t) == int
        t = torch.tensor(t).to(self.dummy.device)

        coeff_noise = self.betas[t]/self.one_minus_alphas_bar_sqrt[t]
        epsi_t = self.noise_net(xt, t)

        drift = torch.sqrt(1/self.alphas[t]) * (xt-coeff_noise*epsi_t)
        sigma_t = torch.sqrt(self.betas[t])
        z = torch.randn_like(xt).to(self.dummy.device)

        x_t_1 = drift + sigma_t*z
        return x_t_1

    def sample(self, x_noise):
        xt = x_noise
        traj = [xt]
        for i in reversed(range(self.diff_steps)):
            xt = self._reverse_one_step(xt, i)
            traj.append(xt)
        #
        return traj



s_curve, _ = make_s_curve(10000, noise=0.1)
s_curve_2d = s_curve[:, [0,2]]/10.0

s_data = torch.Tensor(s_curve_2d).float().to(device)

model = DDPM(
    s_dim=2,
    diff_steps=100,
).to(device)

batch_size = 128
dataloader = torch.utils.data.DataLoader(s_data, batch_size=batch_size,shuffle=True)

max_epochs = 5000
test_freq = 500
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

figs_path = os.path.join('__res__', 'figs')
dict_path = os.path.join('__res__', 'stat_dicts')

create_path(figs_path)
create_path(dict_path)

for ie in tqdm(range(max_epochs)):

    for idx,batch_x in enumerate(dataloader):
        loss = model.eval_noise_loss(batch_x)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
#
    if ie % test_freq == 0:
        print('epcoh={}, loss={:5f}'.format(ie, loss.item()))
        x_noise= torch.randn(s_data.shape).to(device)
        x_seq = model.sample(x_noise)

        fig,axs = plt.subplots(1,10,figsize=(28,3))
        for i in range(1,11):
            cur_x = x_seq[i*10].detach()
            axs[i-1].scatter(cur_x[:,0],cur_x[:,1],color='red',edgecolor='white');
            axs[i-1].set_axis_off();
            axs[i-1].set_title('$q(\mathbf{x}_{'+str(i*10)+'})$')

        plt.savefig(os.path.join('__res__', 'figs', 'epoch{}.png').format(ie), )



