from __future__ import annotations

import pytest
import torch

import flows.transforms

BATCH_SIZE: int = 100
LATTICE_SHAPE: tuple[int] = (8, 8)

TRANSFORMS = [
    flows.transforms.PointwiseAdditiveTransform(),
    flows.transforms.PointwiseAffineTransform(),
    flows.transforms.PointwiseRationalQuadraticSplineTransform(8, (-5, 5)),
]


@pytest.fixture(params=TRANSFORMS)
def transform(request):
    yield request.param


def test_call(transform):
    """Tests that transform called with (in_tensor, params) returns successfully."""
    in_tensor = torch.rand(BATCH_SIZE, 1, *LATTICE_SHAPE)
    params = torch.rand(BATCH_SIZE, transform.params_dof, *LATTICE_SHAPE)

    out_tensor, log_det_jacob = transform(in_tensor, params)

    assert out_tensor.shape == in_tensor.shape
    assert log_det_jacob.shape == torch.Size([out_tensor.shape[0]])


def test_closure(transform):
    """Tests that transforms with inverses round-trip correctly."""
    in_tensor = torch.rand(BATCH_SIZE, 1, *LATTICE_SHAPE)
    params = torch.rand(BATCH_SIZE, transform.params_dof, *LATTICE_SHAPE)

    out_tensor, log_det_jacob_forward = transform(in_tensor, params)
    # TODO: this is wasteful. Think of a better way to skip when inverse NI
    try:
        result, log_det_jacob_inverse = transform.inverse(out_tensor, params)
    except NotImplementedError:
        return

    assert torch.allclose(in_tensor, result, atol=1e-6)  # default atol=1e-8 too low
    assert torch.allclose(log_det_jacob_forward, -log_det_jacob_inverse)


def test_identity(transform):
    """Tests that the identity transformation is as expected."""
    in_tensor = torch.rand(BATCH_SIZE, 1, *LATTICE_SHAPE)
    params = transform.identity_params.view(1, -1, 1, 1).expand(
        BATCH_SIZE, -1, *LATTICE_SHAPE
    )
    out_tensor, log_det_jacob = transform(in_tensor, params)

    assert torch.allclose(in_tensor, out_tensor, atol=1e-7)
    assert torch.allclose(log_det_jacob, torch.zeros_like(log_det_jacob), atol=1e-5)
