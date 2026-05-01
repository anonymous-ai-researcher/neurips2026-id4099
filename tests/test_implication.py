"""Tests for Sigmoidal Reichenbach implication properties."""
import torch
import pytest
from fuzzycell.modules.implication import SigmoidalReichenbach

def test_centering():
    """I_sigma(0.5, 0.5) = 0.5 for all s > 0."""
    for s in [1.0, 5.0, 10.0, 50.0]:
        impl = SigmoidalReichenbach(s=s)
        a = torch.tensor(0.5)
        b = torch.tensor(0.5)
        result = impl(a, b)
        assert abs(result.item() - 0.5) < 1e-6, f"s={s}: got {result.item()}"

def test_gradient_nondegeneracy():
    """Gradient is nonzero everywhere on (0,1)^2."""
    impl = SigmoidalReichenbach(s=10.0)
    a = torch.linspace(0.01, 0.99, 50, requires_grad=True)
    b = torch.linspace(0.01, 0.99, 50, requires_grad=True)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    aa = aa.clone().requires_grad_(True)
    bb = bb.clone().requires_grad_(True)
    result = impl(aa, bb)
    result.sum().backward()
    assert (aa.grad.abs() > 0).all(), "Gradient w.r.t. a is zero somewhere"
    assert (bb.grad.abs() > 0).all(), "Gradient w.r.t. b is zero somewhere"

def test_monotonicity():
    """I_sigma is decreasing in a and increasing in b."""
    impl = SigmoidalReichenbach(s=10.0)
    a1, a2 = torch.tensor(0.3), torch.tensor(0.7)
    b = torch.tensor(0.5)
    assert impl(a1, b) > impl(a2, b), "Not decreasing in a"
    b1, b2 = torch.tensor(0.3), torch.tensor(0.7)
    a = torch.tensor(0.5)
    assert impl(a, b2) > impl(a, b1), "Not increasing in b"

def test_boundary_convergence():
    """As s -> inf, I_sigma converges to classical implication."""
    impl = SigmoidalReichenbach(s=100.0)
    assert impl(torch.tensor(0.8), torch.tensor(0.3)).item() < 0.1  # a > b: should be ~0
    assert impl(torch.tensor(0.3), torch.tensor(0.8)).item() > 0.9  # a < b: should be ~1

if __name__ == "__main__":
    test_centering()
    test_gradient_nondegeneracy()
    test_monotonicity()
    test_boundary_convergence()
    print("All tests passed!")
