# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Tools for the agent section of demo.ipynb.

These are deterministic local functions with fixed data. Nothing here reaches
the network -- the point of the demo is that the whole loop runs on one box.
Swap these for your own functions -- a SaaS API, an internal service, a
database -- and the agent loop is unchanged.
"""

from __future__ import annotations

# --- fixed data --------------------------------------------------------------

_FLIGHTS = {
    ("SFO", "AUS"): [
        {"flight": "NV902", "depart": "16:45", "arrive": "22:20", "fare_usd": 214},
        {"flight": "NV118", "depart": "07:10", "arrive": "12:55", "fare_usd": 268},
    ],
    ("SFO", "JFK"): [
        {"flight": "NV440", "depart": "06:00", "arrive": "14:35", "fare_usd": 331},
        {"flight": "NV441", "depart": "13:20", "arrive": "21:50", "fare_usd": 298},
    ],
    ("AUS", "SFO"): [
        {"flight": "NV903", "depart": "09:30", "arrive": "11:40", "fare_usd": 226},
    ],
}

_RATES = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 156.4, "INR": 83.1, "CAD": 1.36}


# --- implementations ---------------------------------------------------------

def search_flights(origin: str, destination: str, date: str) -> dict:
    """Return available flights for a route on a date."""
    key = (origin.strip().upper(), destination.strip().upper())
    results = _FLIGHTS.get(key, [])
    return {
        "origin": key[0],
        "destination": key[1],
        "date": date,
        "results": results,
        "count": len(results),
    }


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between currencies at fixed reference rates."""
    src = from_currency.strip().upper()
    dst = to_currency.strip().upper()
    if src not in _RATES or dst not in _RATES:
        return {
            "error": f"Unsupported currency pair {src}->{dst}",
            "supported": sorted(_RATES),
        }
    converted = amount / _RATES[src] * _RATES[dst]
    return {
        "amount": amount,
        "from": src,
        "to": dst,
        "rate": round(_RATES[dst] / _RATES[src], 6),
        "converted": round(converted, 2),
    }


# --- dispatch ----------------------------------------------------------------

TOOL_IMPLEMENTATIONS = {
    "search_flights": search_flights,
    "convert_currency": convert_currency,
}


# --- schemas (OpenAI tool-calling format) ------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": (
                "Search available flights between two airports on a given date. "
                "Returns flight numbers, times, and fares in USD."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Origin airport IATA code, e.g. SFO",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination airport IATA code, e.g. AUS",
                    },
                    "date": {
                        "type": "string",
                        "description": "Departure date in YYYY-MM-DD format",
                    },
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert a monetary amount from one currency to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount to convert"},
                    "from_currency": {
                        "type": "string",
                        "description": "Source currency code, e.g. USD",
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "Target currency code, e.g. EUR",
                    },
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
]
