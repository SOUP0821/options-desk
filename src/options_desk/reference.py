"""Numbers the pricer is checked against.

Prices are the QFRM::BS_Simple examples. The first one is Hull example 15.6
(the book rounds the call to 4.76; the extra digits are from QFRM).
The Greek case is Hull's 20-week call, S=49 K=50. His vega and rho are
per point, so compare vega/100 and rho/100.
"""

from __future__ import annotations

PRICE_CASES = (
    {
        "spot": 42.0,
        "strike": 40.0,
        "rate": 0.10,
        "dividend": 0.0,
        "years": 0.5,
        "vol": 0.20,
        "call": 4.759422,
        "put": 0.8085994,
        "source": "Hull example 15.6 / QFRM BS_Simple",
    },
    {
        "spot": 100.0,
        "strike": 90.0,
        "rate": 0.05,
        "dividend": 0.0,
        "years": 2.0,
        "vol": 0.30,
        "call": 26.24017,
        "put": 7.675535,
        "source": "QFRM BS_Simple",
    },
    {
        "spot": 50.0,
        "strike": 60.0,
        "rate": 0.10,
        "dividend": 0.20,
        "years": 3.0,
        "vol": 0.25,
        "call": 0.9941339,
        "put": 18.00265,
        "source": "QFRM BS_Simple, continuous dividend",
    },
    {
        "spot": 90.0,
        "strike": 90.0,
        "rate": 0.15,
        "dividend": 0.0,
        "years": 0.5,
        "vol": 0.20,
        "call": 8.812221,
        "put": 2.309134,
        "source": "QFRM BS_Simple",
    },
    {
        "spot": 15.0,
        "strike": 15.0,
        "rate": 0.01,
        "dividend": 0.0,
        "years": 0.5,
        "vol": 0.50,
        "call": 2.137109,
        "put": 2.062296,
        "source": "QFRM BS_Simple",
    },
)

# Tolerances match the last printed digit of those sources, with a little room.
PRICE_ABSOLUTE_TOLERANCE = 5e-6

HULL_GREEKS = {
    "spot": 49.0,
    "strike": 50.0,
    "rate": 0.05,
    "dividend": 0.0,
    "years": 20 / 52,
    "vol": 0.20,
    "price": 2.40,
    "delta": 0.522,
    "gamma": 0.066,
    "vega_per_point": 0.121,
    "theta_per_year": -4.31,
    "rho_per_point": 0.089,
    "tolerance": {
        "price": 0.005,
        "delta": 0.0015,
        "gamma": 0.0015,
        "vega_per_point": 0.002,
        "theta_per_year": 0.05,
        "rho_per_point": 0.002,
    },
    "source": "Hull Greek-letter example, 20-week European call",
}
