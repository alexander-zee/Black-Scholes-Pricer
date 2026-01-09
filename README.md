# Black–Scholes Option Pricer (Streamlit)

An interactive **European option pricing dashboard** built in Python using the **Black–Scholes model with continuous dividend yield**.

The application prices **calls and puts**, computes **Greeks**, solves for **implied volatility**, and visualizes option prices across **strike and maturity dimensions**.

---

## 🔧 Technologies

- Python  
- NumPy  
- SciPy  
- matplotlib  
- Streamlit  

---

## 📊 Features

This dashboard provides:

- **European Option Pricing**  
  Price call and put options using the Black–Scholes model with continuous dividend yield.

- **Greeks Calculation**  
  Computes:
  - Delta  
  - Gamma  
  - Vega  
  - Theta (per year)  
  - Rho  

- **Implied Volatility Solver**  
  Computes implied volatility from a market option price using a robust bracketing method and Brent’s root-finding algorithm.

- **Price vs Strike Curve**  
  Visualize option prices as a function of the strike price for fixed maturity and volatility.

- **Price Heatmaps**  
  Display call and put price surfaces over grids of:
  - Strike  
  - Time to maturity  

---

## 📐 Model Specification

The Black–Scholes model assumes:

- European exercise
- Lognormal underlying price dynamics
- Constant volatility and interest rates
- Frictionless markets
- Continuous dividend yield \( q \)

### Core quantities:

\[
d_1 = \frac{\ln(S/K) + (r - q + 0.5\sigma^2)T}{\sigma\sqrt{T}}, \quad
d_2 = d_1 - \sigma\sqrt{T}
\]

Option prices are computed using discounted expected payoffs under the risk-neutral measure.

---

## 🧠 Implementation Details

- Pricing and Greeks are implemented analytically.
- Implied volatility is computed by solving:
  
  \[
  BS(\sigma) - P_{market} = 0
  \]

  using Brent’s method (`scipy.optimize.brentq`).
- Heatmaps are generated using vectorized pricing over strike–maturity grids.
- The UI is built with **Streamlit**, separating inputs, outputs, and visual diagnostics.

---

## ▶️ Running the App

### Install dependencies
```bash
pip install streamlit numpy scipy matplotlib
