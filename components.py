import torch
from torch import nn


# https://github.com/bmild/nerf/blob/55d8b00244d7b5178f4d003526ab6667683c9da9/run_nerf_helpers.py#L59
# https://github.com/bmild/nerf/blob/55d8b00244d7b5178f4d003526ab6667683c9da9/run_nerf_helpers.py#L22
class PositionalEncoder(nn.Module):
    def __init__(self, in_nc, n_freq, log_sampling=True, include_input=True):
        super().__init__()

        max_freq = n_freq - 1
        if log_sampling:
            self.freq_bands = 2.**torch.linspace(0., max_freq, n_freq)
        else:
            self.freq_bands = torch.linspace(2.**0., 2.**max_freq, n_freq)

        self.encode_fns = [torch.sin, torch.cos]
        self.out_nc = in_nc * len(self.encode_fns) * n_freq

        self.include_input = include_input
        if self.include_input:
            self.out_nc += in_nc

    def get_output_size(self):
        return self.out_nc

    def forward(self, x):
        """
        Inputs:
            x: (B, C)
        Returns:
            out: (B, C * len(self.encode_fns) * n_freq)
        """
        out = []
        if self.include_input:
            out.append(x)
        for freq in self.freq_bands:
            for fn in self.encode_fns:
                out.append(fn(freq * x))

        out = torch.cat(out, -1)
        return out
