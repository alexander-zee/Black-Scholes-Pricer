import math
from typing import Dict, Tuple

import numpy as np
import streamlit as st
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib.pyplot as plt


def compute_d1_d2(
    spot_price: float,
    strike_price: float,
    time_to_maturity_years: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> Tuple[float, float]:
    """Compute Black-Scholes d1 and d2 with continuous dividend yield."""
    if time_to_maturity_years <= 0 or volatility <= 0 or spot_price <= 0 or strike_price <= 0:
        return float("nan"), float("nan")

    volatility_sqrt_time: float = volatility * math.sqrt(time_to_maturity_years)
    numerator: float = math.log(spot_price / strike_price) + (
        (risk_free_rate - dividend_yield + 0.5 * volatility * volatility) * time_to_maturity_years
    )
    d1: float = numerator / volatility_sqrt_time
    d2: float = d1 - volatility_sqrt_time
    return d1, d2


def black_scholes_price(
    option_type: str,
    spot_price: float,
    strike_price: float,
    time_to_maturity_years: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Price a European call or put using Black-Scholes with continuous dividend yield."""
    option_type_lower: str = option_type.lower()

    if time_to_maturity_years <= 0 or volatility <= 0:
        # discounted intrinsic value fallback
        fwd_spot = spot_price * math.exp(-dividend_yield * time_to_maturity_years)
        disc_strike = strike_price * math.exp(-risk_free_rate * time_to_maturity_years)
        if option_type_lower == "call":
            return max(fwd_spot - disc_strike, 0.0)
        return max(disc_strike - fwd_spot, 0.0)

    d1, d2 = compute_d1_d2(
        spot_price=spot_price,
        strike_price=strike_price,
        time_to_maturity_years=time_to_maturity_years,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )

    disc_spot: float = spot_price * math.exp(-dividend_yield * time_to_maturity_years)
    disc_strike: float = strike_price * math.exp(-risk_free_rate * time_to_maturity_years)

    if option_type_lower == "call":
        return disc_spot * norm.cdf(d1) - disc_strike * norm.cdf(d2)
    return disc_strike * norm.cdf(-d2) - disc_spot * norm.cdf(-d1)


def compute_greeks(
    option_type: str,
    spot_price: float,
    strike_price: float,
    time_to_maturity_years: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> Dict[str, float]:
    """Compute Black-Scholes Greeks with continuous dividend yield."""
    option_type_lower: str = option_type.lower()

    if time_to_maturity_years <= 0 or volatility <= 0 or spot_price <= 0 or strike_price <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1, d2 = compute_d1_d2(
        spot_price=spot_price,
        strike_price=strike_price,
        time_to_maturity_years=time_to_maturity_years,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    disc_spot: float = spot_price * math.exp(-dividend_yield * time_to_maturity_years)
    disc_strike: float = strike_price * math.exp(-risk_free_rate * time_to_maturity_years)

    pdf_d1: float = norm.pdf(d1)
    cdf_d1: float = norm.cdf(d1)
    cdf_minus_d1: float = norm.cdf(-d1)
    cdf_d2: float = norm.cdf(d2)
    cdf_minus_d2: float = norm.cdf(-d2)

    gamma: float = (disc_spot * pdf_d1) / (spot_price * volatility * math.sqrt(time_to_maturity_years))
    vega: float = disc_spot * pdf_d1 * math.sqrt(time_to_maturity_years)

    if option_type_lower == "call":
        delta: float = math.exp(-dividend_yield * time_to_maturity_years) * cdf_d1
        theta: float = (
            -(disc_spot * pdf_d1 * volatility) / (2.0 * math.sqrt(time_to_maturity_years))
            - risk_free_rate * disc_strike * cdf_d2
            + dividend_yield * disc_spot * cdf_d1
        )
        rho: float = strike_price * time_to_maturity_years * math.exp(-risk_free_rate * time_to_maturity_years) * cdf_d2
    else:
        delta = math.exp(-dividend_yield * time_to_maturity_years) * (cdf_d1 - 1.0)
        theta = (
            -(disc_spot * pdf_d1 * volatility) / (2.0 * math.sqrt(time_to_maturity_years))
            + risk_free_rate * disc_strike * cdf_minus_d2
            - dividend_yield * disc_spot * cdf_minus_d1
        )
        rho = -strike_price * time_to_maturity_years * math.exp(-risk_free_rate * time_to_maturity_years) * cdf_minus_d2

    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta), "rho": float(rho)}


def implied_volatility_from_price(
    target_option_price: float,
    option_type: str,
    spot_price: float,
    strike_price: float,
    time_to_maturity_years: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    initial_lower_vol: float = 1e-6,
    initial_upper_vol: float = 5.0,
) -> float:
    """Compute implied volatility using a robust bracketing approach."""
    if target_option_price <= 0:
        raise ValueError("Target option price must be positive for implied volatility.")
    if time_to_maturity_years <= 0 or spot_price <= 0 or strike_price <= 0:
        raise ValueError("Invalid inputs for implied volatility calculation.")

    option_type_lower: str = option_type.lower()

    def price_diff(vol: float) -> float:
        price: float = black_scholes_price(
            option_type=option_type_lower,
            spot_price=spot_price,
            strike_price=strike_price,
            time_to_maturity_years=time_to_maturity_years,
            risk_free_rate=risk_free_rate,
            volatility=vol,
            dividend_yield=dividend_yield,
        )
        return price - target_option_price

    lower_vol: float = initial_lower_vol
    upper_vol: float = initial_upper_vol

    lower_value: float = price_diff(lower_vol)
    upper_value: float = price_diff(upper_vol)

    ceiling_vol: float = 10.0
    while lower_value * upper_value > 0 and upper_vol < ceiling_vol:
        upper_vol *= 1.5
        upper_value = price_diff(upper_vol)

    if lower_value * upper_value > 0:
        raise ValueError("Unable to bracket implied volatility with the provided target price.")

    return float(brentq(price_diff, a=lower_vol, b=upper_vol, maxiter=100, xtol=1e-10))


def compute_price_curve_vs_strike(
    option_type: str,
    spot_price: float,
    time_to_maturity_years: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float,
    min_strike: float,
    max_strike: float,
    num_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute arrays of strikes and corresponding option prices for plotting."""
    strikes: np.ndarray = np.linspace(min_strike, max_strike, num_points)
    prices: np.ndarray = np.array(
        [
            black_scholes_price(
                option_type=option_type,
                spot_price=spot_price,
                strike_price=float(k),
                time_to_maturity_years=time_to_maturity_years,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
            )
            for k in strikes
        ]
    )
    return strikes, prices


def compute_price_heatmap(
    option_type: str,
    spot_price: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
) -> np.ndarray:
    """
    Return a matrix M of shape (len(maturities), len(strikes)),
    where M[i,j] = option price at T=maturities[i], K=strikes[j].
    """
    M = np.empty((len(maturities), len(strikes)), dtype=float)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            M[i, j] = black_scholes_price(
                option_type=option_type,
                spot_price=spot_price,
                strike_price=float(K),
                time_to_maturity_years=float(T),
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
            )
    return M


def format_percent(x: float) -> str:
    return f"{x * 100:.4f}%"


def main() -> None:
    st.set_page_config(page_title="Black-Scholes Pricer (q-dividend)", layout="wide")
    st.title("European Option Pricer – Black-Scholes with Dividend Yield")
    st.caption("Prices European calls/puts, shows Greeks, plots price vs strike, IV, and heatmaps.")

    with st.sidebar:
        st.header("Inputs")
        spot_price: float = st.number_input("Spot price S", min_value=0.0, value=100.0, step=0.1, format="%.6f")
        strike_price: float = st.number_input("Strike price K", min_value=0.0, value=100.0, step=0.1, format="%.6f")
        time_to_maturity_years: float = st.number_input(
            "Time to maturity T (years)", min_value=0.0, value=1.0, step=0.01, format="%.6f"
        )
        risk_free_rate: float = st.number_input(
            "Risk-free rate r (decimal)", min_value=-1.0, value=0.05, step=0.001, format="%.6f"
        )
        dividend_yield: float = st.number_input(
            "Dividend yield q (decimal)", min_value=0.0, value=0.0, step=0.001, format="%.6f"
        )
        volatility: float = st.number_input(
            "Volatility sigma (decimal)", min_value=0.0, value=0.2, step=0.001, format="%.6f"
        )
        option_type: str = st.radio("Option type", ["Call", "Put"], index=0, horizontal=True)

        st.subheader("Implied Volatility")
        market_price: float = st.number_input(
            "Market option price (same type)", min_value=0.0, value=0.0, step=0.01, format="%.6f"
        )

        st.subheader("Plot Settings (Curve)")
        strike_range_low_mult: float = st.slider("Min strike multiplier", 0.1, 1.0, 0.5, 0.05)
        strike_range_high_mult: float = st.slider("Max strike multiplier", 1.0, 3.0, 1.5, 0.05)
        num_points: int = st.slider("Points on curve", 20, 400, 150, 10)

        st.subheader("Heatmap Settings")
        hm_strike_low_mult: float = st.slider("Heatmap: min strike × spot", 0.1, 1.0, 0.5, 0.05)
        hm_strike_high_mult: float = st.slider("Heatmap: max strike × spot", 1.0, 3.0, 2.0, 0.05)
        hm_T_min: float = st.number_input("Heatmap: min T (years)", min_value=0.0, value=0.05, step=0.01, format="%.4f")
        hm_T_max: float = st.number_input("Heatmap: max T (years)", min_value=0.001, value=2.0, step=0.01, format="%.4f")
        hm_nK: int = st.slider("Heatmap: # strike points", 20, 300, 120, 10)
        hm_nT: int = st.slider("Heatmap: # maturity points", 20, 300, 120, 10)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Theoretical Prices")
        call_price: float = black_scholes_price(
            option_type="call",
            spot_price=spot_price,
            strike_price=strike_price,
            time_to_maturity_years=time_to_maturity_years,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
        )
        put_price: float = black_scholes_price(
            option_type="put",
            spot_price=spot_price,
            strike_price=strike_price,
            time_to_maturity_years=time_to_maturity_years,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
        )
        st.metric("Call price", f"{call_price:.6f}")
        st.metric("Put price", f"{put_price:.6f}")

        st.divider()
        st.subheader("Greeks (selected type)")
        greeks: Dict[str, float] = compute_greeks(
            option_type=option_type,
            spot_price=spot_price,
            strike_price=strike_price,
            time_to_maturity_years=time_to_maturity_years,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
        )
        st.write(
            {
                "Delta": round(greeks["delta"], 6),
                "Gamma": round(greeks["gamma"], 6),
                "Vega (per 1.0 vol)": round(greeks["vega"], 6),
                "Theta (per year)": round(greeks["theta"], 6),
                "Rho (per 1.0 rate)": round(greeks["rho"], 6),
            }
        )

    with col_right:
        st.subheader("Price vs Strike")
        min_strike: float = max(1e-8, strike_price * strike_range_low_mult)
        max_strike: float = max(min_strike * 1.0001, strike_price * strike_range_high_mult)
        strikes, prices = compute_price_curve_vs_strike(
            option_type=option_type,
            spot_price=spot_price,
            time_to_maturity_years=time_to_maturity_years,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
            min_strike=min_strike,
            max_strike=max_strike,
            num_points=num_points,
        )
        st.line_chart({"strike": strikes, "price": prices}, x="strike", y="price")

        st.divider()
        st.subheader("Implied Volatility (from market price)")
        if market_price > 0:
            try:
                implied_vol: float = implied_volatility_from_price(
                    target_option_price=market_price,
                    option_type=option_type,
                    spot_price=spot_price,
                    strike_price=strike_price,
                    time_to_maturity_years=time_to_maturity_years,
                    risk_free_rate=risk_free_rate,
                    dividend_yield=dividend_yield,
                )
                st.metric("Implied volatility (decimal)", f"{implied_vol:.6f}")
                st.caption(f"That is {format_percent(implied_vol)}")
            except Exception as exc:
                st.error(f"Implied volatility could not be computed: {exc}")
        else:
            st.info("Enter a positive market price (sidebar) to compute implied volatility.")

    st.divider()
    st.subheader("Heatmaps: Price vs Strike & Maturity")

    # Build grids for heatmaps (strike × maturity)
    hm_K_min = max(1e-8, spot_price * hm_strike_low_mult)
    hm_K_max = max(hm_K_min * 1.0001, spot_price * hm_strike_high_mult)
    if hm_T_max <= hm_T_min:
        hm_T_max = hm_T_min + 1e-4

    hm_strikes = np.linspace(hm_K_min, hm_K_max, hm_nK)
    hm_maturities = np.linspace(hm_T_min, hm_T_max, hm_nT)

    call_matrix = compute_price_heatmap(
        option_type="call",
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        strikes=hm_strikes,
        maturities=hm_maturities,
    )
    put_matrix = compute_price_heatmap(
        option_type="put",
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        strikes=hm_strikes,
        maturities=hm_maturities,
    )

    # Show the two heatmaps side-by-side
    hcol1, hcol2 = st.columns(2, gap="large")
    with hcol1:
        st.markdown("**Call Price Heatmap**")
        fig1, ax1 = plt.subplots(figsize=(6.5, 4.8))
        im1 = ax1.imshow(
            call_matrix,
            aspect="auto",
            origin="lower",
            extent=[hm_strikes.min(), hm_strikes.max(), hm_maturities.min(), hm_maturities.max()],
        )
        ax1.set_xlabel("Strike")
        ax1.set_ylabel("Maturity (years)")
        ax1.set_title("Call Price")
        cbar1 = fig1.colorbar(im1, ax=ax1)
        cbar1.set_label("Price")
        st.pyplot(fig1)

    with hcol2:
        st.markdown("**Put Price Heatmap**")
        fig2, ax2 = plt.subplots(figsize=(6.5, 4.8))
        im2 = ax2.imshow(
            put_matrix,
            aspect="auto",
            origin="lower",
            extent=[hm_strikes.min(), hm_strikes.max(), hm_maturities.min(), hm_maturities.max()],
        )
        ax2.set_xlabel("Strike")
        ax2.set_ylabel("Maturity (years)")
        ax2.set_title("Put Price")
        cbar2 = fig2.colorbar(im2, ax=ax2)
        cbar2.set_label("Price")
        st.pyplot(fig2)

    st.divider()
    st.caption("Notes: Heatmaps display *theoretical* Black–Scholes prices as a function of strike and maturity for the inputs above.")


if __name__ == "__main__":
    main()
