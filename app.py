import math
from typing import Dict, Tuple

import numpy as np
import streamlit as st
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib.pyplot as plt


# ---------- Black–Scholes core ----------
def compute_d1_d2(
    spot_price: float,
    strike_price: float,
    time_to_maturity_years: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> Tuple[float, float]:
    if time_to_maturity_years <= 0 or volatility <= 0 or spot_price <= 0 or strike_price <= 0:
        return float("nan"), float("nan")

    v_sqrt_t: float = volatility * math.sqrt(time_to_maturity_years)
    num: float = math.log(spot_price / strike_price) + (
        (risk_free_rate - dividend_yield + 0.5 * volatility * volatility) * time_to_maturity_years
    )
    d1: float = num / v_sqrt_t
    d2: float = d1 - v_sqrt_t
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
    ot = option_type.lower()
    if time_to_maturity_years <= 0 or volatility <= 0:
        fwd_spot = spot_price * math.exp(-dividend_yield * time_to_maturity_years)
        disc_k = strike_price * math.exp(-risk_free_rate * time_to_maturity_years)
        return max(fwd_spot - disc_k, 0.0) if ot == "call" else max(disc_k - fwd_spot, 0.0)

    d1, d2 = compute_d1_d2(spot_price, strike_price, time_to_maturity_years, risk_free_rate, volatility, dividend_yield)
    disc_spot = spot_price * math.exp(-dividend_yield * time_to_maturity_years)
    disc_k = strike_price * math.exp(-risk_free_rate * time_to_maturity_years)
    return disc_spot * norm.cdf(d1) - disc_k * norm.cdf(d2) if ot == "call" else disc_k * norm.cdf(-d2) - disc_spot * norm.cdf(-d1)


def compute_greeks(
    option_type: str,
    spot_price: float,
    strike_price: float,
    time_to_maturity_years: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> Dict[str, float]:
    if time_to_maturity_years <= 0 or volatility <= 0 or spot_price <= 0 or strike_price <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1, d2 = compute_d1_d2(spot_price, strike_price, time_to_maturity_years, risk_free_rate, volatility, dividend_yield)
    disc_spot = spot_price * math.exp(-dividend_yield * time_to_maturity_years)
    disc_k = strike_price * math.exp(-risk_free_rate * time_to_maturity_years)

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_m_d1 = norm.cdf(-d1)
    cdf_d2 = norm.cdf(d2)
    cdf_m_d2 = norm.cdf(-d2)

    gamma = (disc_spot * pdf_d1) / (spot_price * volatility * math.sqrt(time_to_maturity_years))
    vega = disc_spot * pdf_d1 * math.sqrt(time_to_maturity_years)

    if option_type.lower() == "call":
        delta = math.exp(-dividend_yield * time_to_maturity_years) * cdf_d1
        theta = (-(disc_spot * pdf_d1 * volatility) / (2.0 * math.sqrt(time_to_maturity_years))
                 - risk_free_rate * disc_k * cdf_d2
                 + dividend_yield * disc_spot * cdf_d1)
        rho = strike_price * time_to_maturity_years * math.exp(-risk_free_rate * time_to_maturity_years) * cdf_d2
    else:
        delta = math.exp(-dividend_yield * time_to_maturity_years) * (cdf_d1 - 1.0)
        theta = (-(disc_spot * pdf_d1 * volatility) / (2.0 * math.sqrt(time_to_maturity_years))
                 + risk_free_rate * disc_k * cdf_m_d2
                 - dividend_yield * disc_spot * cdf_m_d1)
        rho = -strike_price * time_to_maturity_years * math.exp(-risk_free_rate * time_to_maturity_years) * cdf_m_d2

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
    if target_option_price <= 0:
        raise ValueError("Target option price must be positive.")
    if time_to_maturity_years <= 0 or spot_price <= 0 or strike_price <= 0:
        raise ValueError("Invalid inputs for IV.")

    ot = option_type.lower()

    def f(vol: float) -> float:
        return black_scholes_price(ot, spot_price, strike_price, time_to_maturity_years, risk_free_rate, vol, dividend_yield) - target_option_price

    lo, hi = initial_lower_vol, initial_upper_vol
    f_lo, f_hi = f(lo), f(hi)
    ceiling = 10.0
    while f_lo * f_hi > 0 and hi < ceiling:
        hi *= 1.5
        f_hi = f(hi)
    if f_lo * f_hi > 0:
        raise ValueError("Unable to bracket implied vol.")
    return float(brentq(f, a=lo, b=hi, maxiter=100, xtol=1e-10))


# ---------- helpers ----------
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
    strikes = np.linspace(min_strike, max_strike, num_points)
    prices = np.array([
        black_scholes_price(option_type, spot_price, float(k), time_to_maturity_years, risk_free_rate, volatility, dividend_yield)
        for k in strikes
    ])
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
    M = np.empty((len(maturities), len(strikes)), dtype=float)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            M[i, j] = black_scholes_price(option_type, spot_price, float(K), float(T), risk_free_rate, volatility, dividend_yield)
    return M


def add_cell_numbers(ax, data: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float,
                     fmt: str = ".2f", fontsize: int = 9) -> None:
    """
    Annotate each heatmap cell with its value.
    Works with imshow(extent=[xmin, xmax, ymin, ymax]).
    """
    nrows, ncols = data.shape
    dx = (xmax - xmin) / ncols
    dy = (ymax - ymin) / nrows
    x_centers = xmin + dx * (np.arange(ncols) + 0.5)
    y_centers = ymin + dy * (np.arange(nrows) + 0.5)

    # Choose white/black text for contrast
    thresh = 0.5 * (np.nanmax(data) + np.nanmin(data))

    for i, y in enumerate(y_centers):
        for j, x in enumerate(x_centers):
            val = data[i, j]
            ax.text(
                x, y, format(val, fmt),
                ha="center", va="center",
                fontsize=fontsize,
                color=("white" if val >= thresh else "black"),
            )


def format_percent(x: float) -> str:
    return f"{x * 100:.4f}%"


# ---------- Streamlit UI ----------
def main() -> None:
    st.set_page_config(page_title="Black-Scholes Pricer (q-dividend)", layout="wide")
    st.title("European Option Pricer – Black–Scholes with Dividend Yield")
    st.caption("Prices European calls/puts, shows Greeks, curve, IV, and **numbered heatmaps**.")

    with st.sidebar:
        st.header("Inputs")
        S = st.number_input("Spot price S", min_value=0.0, value=100.0, step=0.1, format="%.6f")
        K0 = st.number_input("Strike price K (for single-price/Greeks)", min_value=0.0, value=100.0, step=0.1, format="%.6f")
        T = st.number_input("Time to maturity T (years)", min_value=0.0, value=1.0, step=0.01, format="%.6f")
        r = st.number_input("Risk-free rate r (decimal)", min_value=-1.0, value=0.05, step=0.001, format="%.6f")
        q = st.number_input("Dividend yield q (decimal)", min_value=0.0, value=0.0, step=0.001, format="%.6f")
        sigma = st.number_input("Volatility σ (decimal)", min_value=0.0, value=0.2, step=0.001, format="%.6f")
        opt_type = st.radio("Option type (for curve/IV/Greeks)", ["Call", "Put"], index=0, horizontal=True)

        st.subheader("Implied Volatility")
        mkt_px = st.number_input("Market option price", min_value=0.0, value=0.0, step=0.01, format="%.6f")

        st.subheader("Curve Settings")
        k_low_mult = st.slider("Min strike multiplier", 0.1, 1.0, 0.5, 0.05)
        k_high_mult = st.slider("Max strike multiplier", 1.0, 3.0, 1.5, 0.05)
        n_curve = st.slider("Points on curve", 20, 400, 150, 10)

        st.subheader("Heatmap Settings")
        # ↓ fewer points -> bigger squares
        hm_nK = st.slider("Heatmap: # strike points (columns)", 10, 120, 30, 5)
        hm_nT = st.slider("Heatmap: # maturity points (rows)", 10, 120, 30, 5)
        hm_K_low = st.slider("Heatmap: min strike × spot", 0.1, 1.0, 0.5, 0.05)
        hm_K_high = st.slider("Heatmap: max strike × spot", 1.0, 3.0, 2.0, 0.05)
        hm_T_min = st.number_input("Heatmap: min T (years)", min_value=0.0, value=0.05, step=0.01, format="%.4f")
        hm_T_max = st.number_input("Heatmap: max T (years)", min_value=0.001, value=2.0, step=0.01, format="%.4f")

        st.subheader("Cell Numbers")
        annotate = st.checkbox("Show numbers in cells", value=True)
        decimals = st.slider("Decimals in cells", 0, 4, 2, 1)
        cell_font = st.slider("Cell font size", 6, 20, 11, 1)
        max_cells = st.slider("Max cells to annotate (perf cap)", 100, 4000, 1600, 100)

    # --- Left column: single prices & Greeks ---
    left, right = st.columns(2)
    with left:
        st.subheader("Theoretical Prices")
        c_px = black_scholes_price("call", S, K0, T, r, sigma, q)
        p_px = black_scholes_price("put",  S, K0, T, r, sigma, q)
        st.metric("Call price", f"{c_px:.6f}")
        st.metric("Put price",  f"{p_px:.6f}")

        st.divider()
        st.subheader("Greeks (selected type)")
        greeks = compute_greeks(opt_type, S, K0, T, r, sigma, q)
        st.write({k.capitalize(): round(v, 6) for k, v in greeks.items()})

    # --- Right column: curve & IV ---
    with right:
        st.subheader("Price vs Strike")
        k_min = max(1e-8, K0 * k_low_mult)
        k_max = max(k_min * 1.0001, K0 * k_high_mult)
        strikes, prices = compute_price_curve_vs_strike(opt_type, S, T, r, sigma, q, k_min, k_max, n_curve)
        st.line_chart({"strike": strikes, "price": prices}, x="strike", y="price")

        st.divider()
        st.subheader("Implied Volatility (from market price)")
        if mkt_px > 0:
            try:
                iv = implied_volatility_from_price(mkt_px, opt_type, S, K0, T, r, q)
                st.metric("Implied vol (decimal)", f"{iv:.6f}")
                st.caption(f"That is {format_percent(iv)}")
            except Exception as e:
                st.error(f"IV could not be computed: {e}")
        else:
            st.info("Enter a positive market price to compute IV.")

    # --- Heatmaps ---
    st.divider()
    st.subheader("Heatmaps: Price vs Strike & Maturity")

    # build grids
    K_min = max(1e-8, S * hm_K_low)
    K_max = max(K_min * 1.0001, S * hm_K_high)
    if hm_T_max <= hm_T_min:
        st.warning("Max T must be greater than Min T. Adjusted automatically.")
        hm_T_max = hm_T_min + 1e-4

    hm_strikes = np.linspace(K_min, K_max, hm_nK)
    hm_mats = np.linspace(hm_T_min, hm_T_max, hm_nT)

    call_M = compute_price_heatmap("call", S, r, sigma, q, hm_strikes, hm_mats)
    put_M  = compute_price_heatmap("put",  S, r, sigma, q, hm_strikes, hm_mats)

    # plotting helper (bigger figure -> bigger squares)
    fmt_str = f".{decimals}f"
    annotate_ok = annotate and (hm_nK * hm_nT <= max_cells)

    c1, c2 = st.columns(2, gap="large")
    for title, M, col in [("Call Price", call_M), ("Put Price", put_M)]:
        with (c1 if title.startswith("Call") else c2):
            # Scale figure size dynamically to keep squares large
            fig_width = hm_nK * 0.25   # 0.25 inches per column
            fig_height = hm_nT * 0.25  # 0.25 inches per row
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=120)
            im = ax.imshow(
                M,
                aspect="auto",
                origin="lower",
                extent=[hm_strikes.min(), hm_strikes.max(), hm_mats.min(), hm_mats.max()],
                interpolation="nearest",  # crisper squares
            )
            ax.set_xlabel("Strike")
            ax.set_ylabel("Maturity (years)")
            ax.set_title(title)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Price")

            if annotate_ok:
                add_cell_numbers(ax, M, hm_strikes.min(), hm_strikes.max(), hm_mats.min(), hm_mats.max(), fmt=fmt_str, fontsize=cell_font)
            elif annotate and not annotate_ok:
                st.caption(f"⚠️ Not annotating: {hm_nK*hm_nT} cells > cap {max_cells}. Lower #points or raise cap.")

            st.pyplot(fig)

    st.caption("Tip: reduce the heatmap #points to make each square larger. Use the cell font size & decimals to tweak readability.")


if __name__ == "__main__":
    main()
