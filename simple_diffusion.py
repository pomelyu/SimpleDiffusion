# Modified from https://github.com/cloneofsimo/minDiffusion/blob/master/superminddpm.py for practice
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid, save_image
from tqdm import tqdm, trange

from components import PositionalEncoder

def train_mnist():
    trainer = DDPMTrainer()
    trainer.train()

class DDPMTrainer():
    def __init__(self, n_epoh=20, save_folder="./ckpt", T=200) -> None:
        self.n_epoch = n_epoh
        self.save_folder = Path(save_folder)
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.T = T

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eps_model = DummyEpsModel(n_channels=1, n_freq=4).to(self.device)
        self.beta_schedule = get_ddpm_beta_schedule(beta1=1e-4, beta2=0.02, T=T).to(self.device)
        self.alpha_schedule = 1 - self.beta_schedule                  # alpha = 1 - beta
        self.alpha_bar_schedule = torch.cumprod(self.alpha_schedule, dim=0)  # alpha_bar_t = alpha_t * alpha_t-1 * ... * alpha_0

        self.dataloader = get_mnist_dataloader(batch_size=64)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=2.e-4)

    def train(self):
        for epoch in trange(self.n_epoch):
            pbar = tqdm(self.dataloader, total=len(self.dataloader))
            loss_ema = None
            for x in pbar:
                # x = (image, label)
                x = x[0]
                loss = self.train_step(x)
                loss_ema = loss if loss_ema is None else loss_ema * 0.9 + loss * 0.1
                pbar.set_description(f"loss: {loss_ema:.4f}")
            self.valid_epoch(epoch, x, B=32)


    def train_step(self, x):
        self.eps_model.train()

        x = x.to(self.device)
        B = len(x)

        # Calculate the loss:
        # eps = Gaussian(0, 1)
        # x_t = sqrt(alpha_bar) * x_t-1 + sqrt(1 - alpha_bar_t) * eps
        # eps_pred = eps_model(x_t, t)
        # loss = Loss(eps, eps_pred)

        # Randomly select beta from beta_schedule and calculate alpha_bar
        ts = torch.randint(1, self.T, size=(B,)).to(self.device)
        alpha_bar = self.alpha_bar_schedule[ts.long()]

        # calculate xt
        eps = torch.rand_like(x)
        x_t = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * eps

        # calculate eps_pred and loss
        eps_pred = self.eps_model(x_t, ts[:, None] / self.T)
        loss = self.criterion(eps, eps_pred)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def valid_epoch(self, epoch, x, B=16, T=15):
        # evalulation
        # z = Gaussian(0, 1) if t > 1 else 0
        # x_t-1 = 1/sqrt(alpha) * (x_t - eps_model(x_t, t) * (1 - alpha)/sqrt(1 - alpha_bar)) + sqrt(beta) * z
        self.eps_model.eval()
        xh = torch.rand(B, *x.shape[1:]).to(self.device)
        ts = torch.arange(T, 0, -1).to(self.device)
        for t in tqdm(ts, total=len(ts)):
            z = torch.rand_like(xh) if t > 1 else 0
            t = t[None].expand(B)
            eps = self.eps_model(xh, t[:, None] / T)
            beta = self.beta_schedule[t.long()]
            alpha = self.alpha_schedule[t.long()]
            alpha_bar = self.alpha_bar_schedule[t.long()]
            xh = 1 / torch.sqrt(alpha_bar) * (xh - eps * (1 - alpha) / torch.sqrt(1 - alpha_bar)) + torch.sqrt(beta) * z

        grid = make_grid(xh, nrow=4)
        save_image(grid, f"{self.save_folder}/ddpm_sample_{epoch}.png")

        # save model
        torch.save(self.eps_model.state_dict(), f"{self.save_folder}/ddpm_mnist_{epoch:0>4d}.pth")


def get_ddpm_beta_schedule(beta1: float, beta2: float, T: int) -> Tensor:
    """ Get the beta schedule value for training
    - q(x_t|x_t-1) = Gaussian(sqrt(1 - beta_t) * x_t-1, beta_t)
    - beta is the noise level from beta1(t=0, real image) to beta2(t=T, white noise)
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    beta_t = beta_t[:, None, None, None]
    return beta_t


conv_block = lambda in_nc, out_nc: nn.Sequential(
    nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(out_nc),
    nn.LeakyReLU(),
)

class DummyEpsModel(nn.Module):
    """ Eps_theta
    - predict noise(eps_i) by the noisy input(x_i) and timestamp(t_i)
    - this model should be a UNet-like structure. For simplicity, just let the input and output dimension the same 
    """
    def __init__(self, n_channels=3, n_freq=4) -> None:
        super().__init__()

        self.encoding = PositionalEncoder(in_nc=1, n_freq=n_freq)
        encoded_nc = self.encoding.get_output_size()

        self.model = nn.Sequential(
            conv_block(n_channels + encoded_nc, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64),
            nn.Conv2d(64, n_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        H, W = x.shape[-2:]
        emb_t = self.encoding(t)
        emb_t = emb_t[:, :, None, None].expand(-1, -1, H, W)
        eps = self.model(torch.cat([x, emb_t], dim=1))
        return eps

def get_mnist_dataloader(data_path="./data", batch_size=64):
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )

    dataset = MNIST(
        data_path,
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    return dataloader


if __name__ == "__main__":
    train_mnist()
