from __future__ import annotations

import torch

import flows.utils as utils
from flows.utils import laplacian_2d

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

Tensor: TypeAlias = torch.Tensor
BoolTensor: TypeAlias = torch.BoolTensor
Module: TypeAlias = torch.nn.Module
IterableDataset: TypeAlias = torch.utils.data.IterableDataset

class CouplingLayer(Module):
    def __init__(self, transform, net_spec: dict):
        super().__init__()
        self.transform = transform
        self.net_a = self.build_convnet(**net_spec)
        self.net_b = self.build_convnet(**net_spec)

    def build_convnet(
        self,
        hidden_shape: tuple[PositiveInt],
        activation: Module = torch.nn.Tanh(),
        final_activation: Module = torch.nn.Identity(),
        kernel_size: PositiveInt = 3,
        use_bias: bool = True,
    ):
        net_shape = [1, *hidden_shape, self.transform.params_dof]
        activations = [activation for _ in hidden_shape] + [final_activation]

        net = []
        for in_channels, out_channels, activation in zip(
            net_shape[:-1], net_shape[1:], activations
        ):
            convolution = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=int( (kernel_size - 1) / 2),
                padding_mode="circular",
                stride=1,
                bias=use_bias,
            )
            net.append(convolution)
            net.append(activation)

        return torch.nn.Sequential(*net)

    def forward(self, x_full: Tensor, log_det_jacob: tensor) -> tuple[Tensor]:
        n_batch, n_channels, l1, l2 = x_full.shape
        if n_channels > 1:
            x, h = torch.tensor_split(x_full, [1], dim=1)  # take first channel
        else:
            x = x_full
        mask = utils.make_checkerboard((l1, l2)).to(x.device)
        mask_expanded = mask.view(1, 1, l1, l2)

        x_a = x[..., mask]
        x_b = x[..., ~mask]

        params_a = self.net_b(x.mul(~mask_expanded))[..., mask]
        y_a, log_det_jacob_a = self.transform(x_a, params_a)

        xy = torch.zeros_like(x)
        xy[..., mask] = y_a

        params_b = self.net_a(xy)[..., ~mask]
        y_b, log_det_jacob_b = self.transform(x_b, params_b)

        y = xy.clone()
        y[..., ~mask] = y_b

        if n_channels > 1:
            y_full = torch.cat([y, h], dim=1)
        else:
            y_full = y

        log_det_jacob.add_(log_det_jacob_a)
        log_det_jacob.add_(log_det_jacob_b)

        return y_full, log_det_jacob


    def inverse(self, y_full: Tensor, log_det_jacob: tensor) -> tuple[Tensor]:
        n_batch, n_channels, l1, l2 = y_full.shape
        if n_channels > 1:
            y, h = torch.tensor_split(y_full, [1], dim=1)  # take first channel
        else:
            y = y_full
        mask = utils.make_checkerboard((l1, l2)).to(y.device)
        mask_expanded = mask.view(1, 1, l1, l2)

        y_a = y[..., mask]
        y_b = y[..., ~mask]

        params_b = self.net_a(y.mul(mask_expanded))[..., ~mask]
        x_b, log_det_jacob_b = self.transform.inverse(y_b, params_b)

        yx = torch.zeros_like(y)
        yx[..., ~mask] = x_b

        params_a = self.net_b(yx)[..., mask]
        x_a, log_det_jacob_a = self.transform.inverse(y_a, params_a)

        x = yx.clone()
        x[..., mask] = x_a

        if n_channels > 1:
            x_full = torch.cat([x, h], dim=1)
        else:
            x_full = x

        log_det_jacob.add_(log_det_jacob_b)
        log_det_jacob.add_(log_det_jacob_a)

        return x_full, log_det_jacob


class UpsamplingLayer(Module):
    def __init__(self, use_batch_dimension: bool = False):
        super().__init__()
        self.use_batch_dimension = use_batch_dimension

        kernel = torch.stack(
            [
                Tensor([[1, 0], [0, 0]]),
                Tensor([[0, 1], [0, 0]]),
                Tensor([[0, 0], [1, 0]]),
                Tensor([[0, 0], [0, 1]]),
            ],
            dim=0,
        ).unsqueeze(dim=1)
        assert kernel.shape == torch.Size([4, 1, 2, 2])

        self.register_buffer("kernel", kernel)

    def forward(self, x: Tensor, log_det_jacob: Tensor) -> tuple[Tensor]:
        """Upsample 1 lattice site -> 4 lattice sites."""
        n_batch, _, l1, l2 = x.shape
        assert (l1 % 2 == 0) and (l2 % 2 == 0)

        y = F.conv_transpose2d(x.view(-1, 4, l1, l2), self.kernel, stride=2)

        if self.use_batch_dimension:
            y = y.view(n_batch // 4, -1, 2 * l1, 2 * l2)
            log_det_jacob = log_det_jacob.view(-1, 4).sum(dim=1)

        else:
            y = y.view(n_batch, -1, 2 * l1, 2 * l2)

        return y, log_det_jacob

    def inverse(self, y: Tensor, log_det_jacob: Tensor) -> tuple[Tensor]:
        """Downsample 4 lattice sites -> 1 lattice site."""
        n_batch, _, l1, l2 = y.shape
        assert (l1 % 2 == 0) and (l2 % 2 == 0)

        x = F.conv2d(y.view(-1, 1, l1, l2), self.kernel, stride=2)

        if self.use_batch_dimension:
            x = x.view(4 * n_batch, -1, l1 // 2, l2 // 2)
            log_det_jacob_upsampled = torch.zeros(4 * log_det_jacob.shape[0], 1)
            log_det_jacob_upsampled[::4] = log_det_jacob
            log_det_jacob = log_det_jacob_upsampled
        else:
            x = x.view(n_batch, -1, l1 // 2, l2 // 2)

        return x, log_det_jacob


_test_input = torch.arange(64).view(1, 1, 8, 8).float()
_test_ldj = torch.zeros([1])
_test_layer = UpsamplingLayer(use_batch_dimension=True)
_test_out1, _test_ldj = _test_layer.inverse(_test_input, _test_ldj)
_test_out2, _test_ldj = _test_layer.inverse(_test_out1, _test_ldj)
assert torch.allclose(_test_out2[0, 0], torch.Tensor([[0, 4], [32, 36]]))
_test_out1_rt, _test_ldj = _test_layer.forward(_test_out2, _test_ldj)
assert torch.allclose(_test_out1, _test_out1_rt)
_test_input_rt, _test_ldj = _test_layer.forward(_test_out1_rt, _test_ldj)
assert torch.allclose(_test_input, _test_input_rt)
assert torch.allclose(_test_ldj, torch.zeros([1]))


class GlobalRescalingLayer(Module):
    def __init__(self):
        super().__init__()
        self.log_scale = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: Tensor, log_det_jacob: Tensor) -> tuple[Tensor]:
        x = x.mul(self.log_scale.exp())
        numel = utils.prod(x.shape[1:])
        log_det_jacob.add_(self.log_scale.mul(numel))
        return x, log_det_jacob

    def inverse(self, y: Tensor, log_det_jacob: Tensor) -> tuple[Tensor]:
        y = y.mul(self.log_scale.neg().exp())
        numel = utils.prod(y.shape[1:])
        log_det_jacob.sub_(self.log_scale.mul(numel))
        return y, log_det_jacob
