import httpx
from fastapi import FastAPI
from typing import Dict
from datetime import datetime
from indicators import (
    fetch_full_klines,
    calculate_ema,
    calculate_rsi,
    calculate_sma,
    calculate_stoch_rsi_kd,
    calculate_volume_sma_and_ratio
)

app = FastAPI()

TIMEFRAMES = {
    "5m": {"interval": "5", "limit": 50},
    "15m": {"interval": "15", "limit": 50},
    "30m": {"interval": "30", "limit": 40},
    "1h": {"interval": "60", "limit": 30},
    "2h": {"interval": "120", "limit": 20},
    "4h": {"interval": "240", "limit": 15}
}

BASE_KLINE_URL = "https://api.bybit.com/v5/market/kline"
BASE_TICKER_URL = "https://api.bybit.com/v5/market/tickers"


# ========== API DATA FETCHERS ==========

async def fetch_kline(symbol: str, interval: str, limit: int) -> Dict:
    params = {
        "category": "spot",
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(BASE_KLINE_URL, params=params)
        response.raise_for_status()
        return response.json()["result"]["list"]


async def fetch_ticker(symbol: str) -> Dict:
    params = {
        "category": "spot",
        "symbol": symbol.upper()
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(BASE_TICKER_URL, params=params)
        response.raise_for_status()
        return response.json()["result"]


# ========== ENDPOINT: RAW MARKET DATA ==========

@app.get("/market/{symbol}")
async def get_market_data(symbol: str):
    klines = {}

    for name, tf in TIMEFRAMES.items():
        candles = await fetch_kline(symbol, tf["interval"], tf["limit"])
        processed = []

        for i, c in enumerate(candles):
            ts = int(c[0])
            entry = {
                "index": i + 1,
                "timestamp": ts,
                "datetime_utc": datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                "open": c[1],
                "high": c[2],
                "low": c[3],
                "close": c[4],
                "volume": c[5],
                "turnover": c[6]
            }
            processed.append(entry)

        klines[name] = processed

    ticker_data = await fetch_ticker(symbol)

    return {
        symbol.upper(): {
            "klines": klines,
            "tickers": ticker_data
        }
    }


# ========== ENDPOINT: TECHNICAL INDICATORS ==========

@app.get("/indicators/{symbol}")
async def get_indicators(symbol: str):
    result = {
        "symbol": symbol.upper()
    }

    for label, tf in TIMEFRAMES.items():
        limit = max(600, tf["limit"] + 200)
        raw_data = await fetch_full_klines(symbol, tf["interval"], limit)

        closes = [float(c[4]) for c in raw_data]
        volumes = [float(c[5]) for c in raw_data]
        timestamps = [int(c[0]) for c in raw_data]

        closes.reverse()
        volumes.reverse()
        timestamps.reverse()

        ema_50 = calculate_ema(closes, 50)[-tf["limit"]:][::-1]
        ema_200 = calculate_ema(closes, 200)[-tf["limit"]:][::-1]

        rsi_full = calculate_rsi(closes, 14)
        rsi = rsi_full[-tf["limit"]:][::-1]
        rsi_ma = calculate_sma(rsi_full, 14)[-tf["limit"]:][::-1]

        stoch_k_full, stoch_d_full = calculate_stoch_rsi_kd(rsi_full, 14, 3, 3)
        stoch_k = stoch_k_full[-tf["limit"]:][::-1]
        stoch_d = stoch_d_full[-tf["limit"]:][::-1]

        vol_sma, vol_ratio = calculate_volume_sma_and_ratio(volumes, 20)
        vol_sma = vol_sma[-tf["limit"]:][::-1]
        vol_ratio = vol_ratio[-tf["limit"]:][::-1]

        indicators = []
        for i in range(tf["limit"]):
            indicators.append({
                "index": i + 1,
                "timestamp": timestamps[-(i + 1)],
                "datetime_utc": datetime.utcfromtimestamp(timestamps[-(i + 1)] / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                "ema_50": round(ema_50[i], 8) if ema_50[i] else None,
                "ema_200": round(ema_200[i], 8) if ema_200[i] else None,
                "rsi": round(rsi[i], 2) if rsi[i] else None,
                "rsi_ma": round(rsi_ma[i], 2) if rsi_ma[i] else None,
                "stoch_k": round(stoch_k[i] * 100, 2) if stoch_k[i] else None,
                "stoch_d": round(stoch_d[i] * 100, 2) if stoch_d[i] else None,
                "volume": volumes[-(i + 1)],
                "volume_sma_20": round(vol_sma[i], 2) if vol_sma[i] else None,
                "volume_ratio": round(vol_ratio[i], 2) if vol_ratio[i] else None
            })

        result[label] = indicators

    return result

from fastapi.responses import FileResponse

@app.get("/privacy", include_in_schema=False)
def privacy_policy():
    return FileResponse("privacy.html", media_type="text/html")

