"""goldlib — shared primitives for the gold research stack."""
from .guards import (
    StaleDataError, ModelError, SampleError, PriceSanityError,
    require_fresh, require_min_obs, require_converged,
    require_price_sane, cap_kelly, manual_field, ManualField,
)
from .prices import MarketContext, get_context, context_from_values
from .gex import compute_gex, put_call_ratio, GexReport, bs_gamma, bs_delta
from .seasonality import intraday_return_profile, verify_timezone
from .scenarios import implied_scenario_probs, ev_table, summarise

__version__ = "1.0.0"
