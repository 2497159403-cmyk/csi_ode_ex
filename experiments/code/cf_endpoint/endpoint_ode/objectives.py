"""Time integration of the original location V_t loss, from public schedule only."""
from __future__ import annotations

# Independently verified by high-order quadrature in tests. No Truth/unknown f/u.
LOCATION_A = 2.607350651390664
LOCATION_C = -1.6948343075966772
LOCATION_D = 3.149918582578128


def integrated_location_loss(model, batch):
    """Integral of ||db R - (da Y0 + db Y1)||², per coordinate.

    Keep the A scale and parameter-independent constant. This only removes
    Monte Carlo time noise, retaining the same empirical endpoints and noise.
    """
    if getattr(model,"variant",None) not in {"location_gated_x_only", "location_gated_fixed_x", "location_cosine_rank1"}:
        raise ValueError("Integrated location loss requires condition-only X")
    residual = model.amplitude(batch.x,batch.y0)
    centered = residual-batch.y1-(LOCATION_C/LOCATION_A)*batch.y0
    return (LOCATION_A*centered.square()+(LOCATION_D-LOCATION_C**2/LOCATION_A)*batch.y0.square()).mean()
