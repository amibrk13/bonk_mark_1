import httpx
import pandas as pd
from typing import List, Tuple

BASE_KLINE_URL = "https://api.bybit.com/v5/market/kline"


async def fetch_full_klines(symbol: str, interval: str, limit: int):
    """Получение свечей с Bybit API для индикаторов"""
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


def calculate_ema(prices: List[float], period: int) -> List[float]:
    return pd.Series(prices).ewm(span=period, adjust=False).mean().tolist()


def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    series = pd.Series(prices)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).tolist()


def calculate_sma(values: List[float], period: int) -> List[float]:
    return pd.Series(values).rolling(window=period, min_periods=period).mean().tolist()


def calculate_stoch_rsi_kd(
    rsi_values: List[float],
    period: int = 14,
    k_period: int = 3,
    d_period: int = 3
) -> Tuple[List[float], List[float]]:
    rsi_series = pd.Series(rsi_values)
    min_val = rsi_series.rolling(window=period).min()
    max_val = rsi_series.rolling(window=period).max()
    stoch_rsi = (rsi_series - min_val) / (max_val - min_val)
    k = stoch_rsi.rolling(window=k_period).mean()
    d = k.rolling(window=d_period).mean()
    return k.tolist(), d.tolist()


def calculate_volume_sma_and_ratio(volumes: List[float], period: int = 20) -> Tuple[List[float], List[float]]:
    volume_series = pd.Series(volumes)
    volume_sma = volume_series.rolling(window=period, min_periods=period).mean()
    volume_ratio = volume_series / volume_sma
    return volume_sma.tolist(), volume_ratio.tolist()
