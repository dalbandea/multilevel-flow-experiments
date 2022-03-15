from __future__ import annotations

import math
import logging

import torch
import torch.nn.functional as F

Tensor: TypeAlias = torch.Tensor
BoolTensor: TypeAlias = torch.BoolTensor
Module: TypeAlias = torch.nn.Module
IterableDataset: TypeAlias = torch.utils.data.IterableDataset
Distribution: TypeAlias = torch.distributions.Distribution

PI = math.pi

logging.basicConfig(level="INFO")
log = logging.getLogger(__name__)


class PointwiseAdditiveTransform:
    """Performs an element-wise translation of the input tensor.

    The transformations are

        x -> y = x + t      (forward)
        y -> x = y - t      (inverse)
    """

    params_dof: int = 1

    @property
    def identity_params(self) -> Tensor:
        return Tensor([0.0])

    def __call__(self, x: Tensor, params: Tensor) -> tuple[Tensor]:
        return self.forward(x, params)

    def forward(self, x: Tensor, shift: Tensor) -> tuple[Tensor]:
        # Don't do in-place operations, just to be safe
        y = x.add(shift)
        log_det_jacob = torch.zeros(x.shape[0]).type_as(x)
        return y, log_det_jacob

    def inverse(self, y: Tensor, shift: Tensor) -> tuple[Tensor]:
        x = y.sub(shift)
        log_det_jacob = torch.zeros(y.shape[0]).type_as(y)
        return x, log_det_jacob


class PointwiseAffineTransform:
    """Performs an element-wise affine transformation of the input tensor.

    The transformations are

        x -> y = x . exp(-s) + t       (forward)
        y -> x = (y - t) . exp(s)      (inverse)
    """

    params_dof: int = 2

    @property
    def identity_params(self) -> Tensor:
        return Tensor([0.0, 0.0])

    def __call__(self, x: Tensor, params: Tensor) -> tuple[Tensor]:
        return self.forward(x, params)

    def forward(self, x: Tensor, params: Tensor) -> tuple[Tensor]:
        log_scale, shift = params.split(1, dim=1)
        y = x.mul(log_scale.neg().exp()).add(shift)
        log_det_jacob = log_scale.neg().flatten(start_dim=1).sum(dim=1)
        return y, log_det_jacob

    def inverse(self, y: Tensor, params: Tensor) -> tuple[Tensor]:
        log_scale, shift = params.split(1, dim=1)
        x = y.sub(shift).mul(log_scale.exp())
        log_det_jacob = log_scale.flatten(start_dim=1).sum(dim=1)
        return x, log_det_jacob


class PointwiseRationalQuadraticSplineTransform:
    """Rational quadratic spline transformation."""

    def __init__(self, n_segments: int, interval: tuple[int]):
        self.n_segments = n_segments
        self.params_dof = 3 * n_segments - 1
        self.interval = interval
        self.lower_boundary = interval[0]
        self.upper_boundary = interval[1]
        self.interval_size = self.upper_boundary - self.lower_boundary
        self.pad_derivs = lambda derivs: F.pad(derivs, (1, 1), "constant", 1)

    def __call__(self, inputs: Tensor, params: Tensor) -> tuple[Tensor]:
        return self.forward(inputs, params)

    @property
    def identity_params(self):
        """Parameters required for spline to enact an identity transform."""
        return torch.cat(
            (
                torch.full(size=(2 * self.n_segments,), fill_value=1 / self.n_segments),
                (
                    torch.ones(self.params_dof - 2 * self.n_segments).exp() - 1
                ).log(),  # one after softmax
            ),
            dim=0,
        )

    def _build_spline(
        self, inputs: Tensor, params: Tensor, inverse: bool = False
    ) -> tuple[Tensor]:
        # Split inner dim into (n_segments, n_segments, n_segments - 1)
        heights, widths, derivs = params.tensor_split(
            (self.n_segments, 2 * self.n_segments), dim=-1
        )
        heights = F.softmax(heights, dim=-1) * self.interval_size
        widths = F.softmax(widths, dim=-1) * self.interval_size
        derivs = self.pad_derivs(F.softplus(derivs))

        knots_xcoords = (
            torch.cat(
                (
                    torch.zeros_like(inputs),
                    torch.cumsum(widths, dim=-1),
                ),
                dim=-1,
            )
            + self.lower_boundary
        )
        knots_ycoords = (
            torch.cat(
                (
                    torch.zeros_like(inputs),
                    torch.cumsum(heights, dim=-1),
                ),
                dim=-1,
            )
            + self.lower_boundary
        )

        if inverse:
            bins = knots_ycoords
        else:
            bins = knots_xcoords
        segment_idx = (
            torch.searchsorted(
                bins,
                inputs,
            )
            - 1
        ).clamp(0, self.n_segments - 1)

        width_this_segment = torch.gather(widths, -1, segment_idx)
        height_this_segment = torch.gather(heights, -1, segment_idx)
        slope_this_segment = height_this_segment / width_this_segment
        # derivs.mul_(slope_this_segment.unsqueeze(dim=-1))  # maybe useful for learning identity transf but not really otherwise
        deriv_at_lower_knot = torch.gather(derivs, -1, segment_idx)
        deriv_at_upper_knot = torch.gather(derivs, -1, segment_idx + 1)
        xcoord_at_lower_knot = torch.gather(knots_xcoords, -1, segment_idx)
        ycoord_at_lower_knot = torch.gather(knots_ycoords, -1, segment_idx)

        return (
            width_this_segment,
            height_this_segment,
            slope_this_segment,
            deriv_at_lower_knot,
            deriv_at_upper_knot,
            xcoord_at_lower_knot,
            ycoord_at_lower_knot,
        )

    def _check_interval(self, inputs: Tensor) -> Tensor:
        outside_interval_mask = torch.logical_or(
            inputs > self.upper_boundary,
            inputs < self.lower_boundary,
        )
        if outside_interval_mask.sum() > 0.001 * len(inputs.view(-1)):
            log.warning(
                "fraction of spline inputs falling outside interval exceeded 1/1000. "
                + f"Perhaps the interval {self.interval} is too narrow?"
            )
        if abs(self.upper_boundary - inputs.max()) > self.interval_size / 10:
            log.warning(
                "no inputs fell in the upper 10% of the interval. "
                + "Perhaps the interval {self.interval} is too wide?"
            )
        return outside_interval_mask

    def forward(self, x: Tensor, params: Tensor) -> tuple[Tensor]:
        """Applies the 'forward' transformation."""
        x = x.squeeze(dim=1).unsqueeze(dim=-1)  # don't do in-place
        params = params.movedim(1, -1)

        outside_interval_mask = self._check_interval(x)

        (w, h, s, d0, d1, x0, y0) = self._build_spline(x, params)

        alpha = (x - x0) / w
        # NOTE: this clamping will hide bugs that result in alpha < 0 or alpha > 1 ...
        alpha.clamp_(0, 1)
        denominator_recip = torch.reciprocal(
            s + (d1 + d0 - 2 * s) * alpha * (1 - alpha)
        )
        beta = (s * alpha.pow(2) + d0 * alpha * (1 - alpha)) * denominator_recip
        y = y0 + h * beta

        gradient = (
            s.pow(2)
            * (
                d1 * alpha.pow(2)
                + 2 * s * alpha * (1 - alpha)
                + d0 * (1 - alpha).pow(2)
            )
            * denominator_recip.pow(2)
        )
        # assert torch.all(gradient > 0)
        gradient[outside_interval_mask] = 1
        log_det_jacob = gradient.log().flatten(start_dim=1).sum(dim=1)

        y[outside_interval_mask] = x[outside_interval_mask]

        y.squeeze_(dim=-1).unsqueeze_(dim=1)

        return y, log_det_jacob

    def inverse(self, y: Tensor, params: Tensor) -> tuple[Tensor]:
        y = y.squeeze(dim=1).unsqueeze(dim=-1)
        params = params.movedim(1, -1)

        outside_interval_mask = self._check_interval(y)

        (w, h, s, d0, d1, x0, y0) = self._build_spline(y, params, inverse=True)

        beta = (y - y0) / h
        beta.clamp_(0, 1)
        b = d0 - (d1 + d0 - 2 * s) * beta
        a = s - b
        c = -s * beta
        alpha = -2 * c * torch.reciprocal(b + (b.pow(2) - 4 * a * c).sqrt())
        x = x0 + w * alpha

        denominator_recip = torch.reciprocal(
            s + (d1 + d0 - 2 * s) * alpha * (1 - alpha)
        )
        gradient_fwd = (
            s.pow(2)
            * (
                d1 * alpha.pow(2)
                + 2 * s * alpha * (1 - alpha)
                + d0 * (1 - alpha).pow(2)
            )
            * denominator_recip.pow(2)
        )
        gradient_fwd[outside_interval_mask] = 1
        log_det_jacob = gradient_fwd.log().flatten(start_dim=1).sum(dim=1).neg()

        x[outside_interval_mask] = y[outside_interval_mask]

        x.squeeze_(dim=-1).unsqueeze_(dim=1)

        return x, log_det_jacob


_test_spline = PointwiseRationalQuadraticSplineTransform(
    n_segments=8, interval=[-PI, PI]
)
_test_x = torch.empty(1000, 1, 4).uniform_(-PI, PI)
_test_params = torch.rand(1000, _test_spline.params_dof, 4) + 0.01
_test_y, _test_ldj = _test_spline.forward(_test_x, _test_params)
_test_x_rt, _test_ldj_inv = _test_spline.inverse(_test_y, _test_params)
assert torch.allclose(_test_x, _test_x_rt, atol=1e-5)
assert torch.allclose(_test_ldj, -_test_ldj_inv, atol=1e-5)
