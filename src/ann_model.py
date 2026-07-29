"""
MindPulse — Module 3  |  Sub-layer 3.2: ANN Architecture
=========================================================
The feedforward MLP that re-ranks Layer 2's output using per-user history
(spec Section 2.3, Sub-layer 3.2).

WHAT IT PREDICTS
----------------
A single scalar per (session, candidate) pair: the predicted RESIDUAL —
how much this user is expected to deviate from the population mean for this
intervention. See `layer3_features.py` for why the residual rather than the
raw rating.

The prediction is used as a correction to Layer 2:

    layer3_score = layer2_score + beta * predicted_residual

with `beta` selected on the training split (see `train_ann.py`). If the model
has learned nothing, its predictions concentrate near zero and Layer 3
degrades gracefully onto Layer 2 rather than injecting noise into the ranking.

SIZED FOR 526 ROWS, NOT FOR THE ARCHITECTURE DIAGRAM
-----------------------------------------------------
There are 526 supervised rows — one per usable training session. The default
23 -> 16 -> 1 network has 401 parameters, already only ~1.3 rows per
parameter. `train_ann.py` therefore sweeps hidden width {8, 16, 32} under
cross-validation rather than fixing it, and applies dropout plus weight decay.

A deliberately small model also matters for interpreting the null result.
Layer 2's experience is the precedent: a validation set too small to measure
AUC reported 0.424 for an architecture that scored 0.653 under proper
cross-validation. An over-parameterised model here would fit user-level noise
and produce exactly the kind of confident-looking output that cannot be
distinguished from genuine personalisation.

EXPECTED BEHAVIOUR ON THIS DATASET
-----------------------------------
Phase 2 measured Ridge at CV R2 = -0.0512 against -0.0022 for predicting the
mean: the target is not predictable from these features. A correctly-behaving
model should therefore COLLAPSE — predictions with near-zero variance. That is
the desired outcome, not a failure, and `train_ann.py` checks for it
explicitly and runs a positive control to prove the trainer can still recover
a signal when one exists.

Author : Module 3 — MindPulse (Team MindForge)
"""

import torch
import torch.nn as nn

IN_DIM     = 23     # must match len(layer3_features.FEATURE_NAMES)
HIDDEN_DIM = 16
DROPOUT    = 0.2


class Layer3ANN(nn.Module):
    """
    Single-hidden-layer MLP regressor over the Layer 3 feature vector.

    Parameters
    ----------
    in_dim : int
        Feature dimensionality; must equal len(FEATURE_NAMES).
    hidden_dim : int
        Width of the hidden layer. Swept under CV in train_ann.py.
    dropout : float
        Applied after the hidden activation; disabled by model.eval().
    """

    def __init__(self, in_dim: int = IN_DIM, hidden_dim: int = HIDDEN_DIM,
                 dropout: float = DROPOUT):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Final layer initialised at zero so an untrained model predicts exactly
        # 0.0 — i.e. "no correction to Layer 2". Any departure from zero is then
        # something training actively produced, which is what makes the collapse
        # check in train_ann.py meaningful rather than an artefact of init.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        """x: (n, in_dim) -> (n,) predicted residuals."""
        return self.net(x).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        return (f"Layer3ANN(\n"
                f"    Linear({self.in_dim} -> {self.hidden_dim})\n"
                f"    ReLU + Dropout({self.dropout})\n"
                f"    Linear({self.hidden_dim} -> 1)\n"
                f"    target     : residual (user deviation from population mean)\n"
                f"    parameters : {self.count_parameters():,}\n"
                f")")


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

def _self_test():
    import numpy as np
    import torch.nn.functional as F

    print("\n" + "=" * 70)
    print("  LAYER 3 — ANN ARCHITECTURE SELF-TEST")
    print("=" * 70)

    torch.manual_seed(42)
    m = Layer3ANN()
    print("\n  Architecture:")
    for line in m.describe().split("\n"):
        print(f"  {line}")

    x = torch.randn(40, IN_DIM)

    print("\n  Shape checks:")
    m.eval()
    with torch.no_grad():
        out = m(x)
    print(f"    Output shape (40,)                 : "
          f"{'PASS' if tuple(out.shape) == (40,) else 'FAIL'}  got {tuple(out.shape)}")
    print(f"    Finite outputs                     : "
          f"{'PASS' if torch.isfinite(out).all() else 'FAIL'}")

    print("\n  Zero-init behaviour:")
    exact_zero = bool((out == 0).all())
    print(f"    Untrained model predicts exactly 0 : "
          f"{'PASS' if exact_zero else 'FAIL'}   (= no correction to Layer 2)")

    print("\n  Gradient flow:")
    m.train()
    loss = F.mse_loss(m(x), torch.randn(40))
    loss.backward()
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    print(f"    All parameters receive gradients    : "
          f"{'PASS' if not missing else 'FAIL ' + str(missing)}")

    print("\n  Dropout behaviour:")
    # The output layer is deliberately zero-initialised, so predictions are
    # exactly 0 no matter what dropout does. Give it non-zero output weights
    # first, or this test silently measures nothing.
    with torch.no_grad():
        m.net[-1].weight.normal_(0, 0.5)
    m.train()
    a, b = m(x), m(x)
    m.eval()
    with torch.no_grad():
        c, d = m(x), m(x)
    print(f"    Stochastic in train mode           : "
          f"{'PASS' if not torch.allclose(a, b) else 'FAIL'}")
    print(f"    Deterministic in eval mode         : "
          f"{'PASS' if torch.allclose(c, d) else 'FAIL'}")

    print("\n  Capacity across the CV sweep:")
    for h in (8, 16, 32):
        mm = Layer3ANN(hidden_dim=h)
        print(f"    hidden={h:<4} parameters={mm.count_parameters():<6} "
              f"rows/param = {526 / mm.count_parameters():.2f}")

    print("\n" + "=" * 70)
    print("  [OK] Architecture verified. Ready for training (Phase 3).")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    _self_test()