import argparse
import warnings
import sys
from datetime import time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONSTANTS & SETTINGS
# ─────────────────────────────────────────────────────────────

NY_TZ = ZoneInfo("America/New_York")

SB_WINDOWS = [
    {"name": "London SB", "short": "LON", "start": time(3, 0), "end": time(4, 0), "color": "#818cf8"},
    {"name": "NY AM SB",  "short": "NYA", "start": time(10, 0), "end": time(11, 0), "color": "#34d399"},
    {"name": "NY PM SB",  "short": "NYP", "start": time(14, 0), "end": time(15, 0), "color": "#fb923c"},
]

DEFAULT_TICKER     = "NQ=F"
DEFAULT_DAYS       = 30
DEFAULT_RR         = 2.0         
DEFAULT_SWING_N    = 3           
DEFAULT_LOOKBACK   = 20          
SL_BUFFER_PCT      = 0.0005      
MIN_FVG_PCT        = 0.0002      
INITIAL_CAPITAL    = 100_000.0
RISK_PER_TRADE_PCT = 1.0         

# Visual Colors
BG, PANEL, PANEL2, GRID_C, TEXT_C, MUTED_C = "#0d1117", "#161b22", "#1c2333", "#21262d", "#e6edf3", "#8b949e"
BULL_C, BEAR_C, WIN_C, LOSS_C, BE_C, EQ_C = "#3fb950", "#f85149", "#34d399", "#f87171", "#fbbf24", "#38bdf8"

# ─────────────────────────────────────────────────────────────
# CORE FUNCTIONS (Data, Detection, Simulation)
# ─────────────────────────────────────────────────────────────

def fetch_5min(ticker: str, days: int) -> pd.DataFrame:
    period = f"{min(days, 60)}d"
    df = yf.download(ticker, interval="5m", period=period, auto_adjust=True, progress=False)
    if df.empty: raise ValueError(f"No data for '{ticker}'")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_convert(NY_TZ) if df.index.tz else pd.to_datetime(df.index).tz_localize("UTC").tz_convert(NY_TZ)
    return df

def find_swing_highs(highs, n=3):
    return np.array([highs[i] == highs[i-n:i+n+1].max() if i >= n and i < len(highs)-n else False for i in range(len(highs))])

def find_swing_lows(lows, n=3):
    return np.array([lows[i] == lows[i-n:i+n+1].min() if i >= n and i < len(lows)-n else False for i in range(len(lows))])

def find_bullish_fvg(df, min_pct=MIN_FVG_PCT):
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    return [(i, h[i-2], l[i]) for i in range(2, len(df)) if l[i] > h[i-2] and (l[i]-h[i-2])/c[i] >= min_pct]

def find_bearish_fvg(df, min_pct=MIN_FVG_PCT):
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    return [(i, h[i], l[i-2]) for i in range(2, len(df)) if l[i-2] > h[i] and (l[i-2]-h[i])/c[i] >= min_pct]

def detect_setups(df, swing_n=DEFAULT_SWING_N, lookback=DEFAULT_LOOKBACK, sl_buffer=SL_BUFFER_PCT):
    setups = []
    for d in sorted(set(df.index.date)):
        day_df = df[df.index.date == d]
        for win in SB_WINDOWS:
            win_mask = [(ts.time() >= win["start"] and ts.time() < win["end"]) for ts in day_df.index]
            if sum(win_mask) < 3: continue
            win_idx = np.where(win_mask)[0][0]
            pre_df = day_df.iloc[max(0, win_idx-lookback):win_idx]
            if pre_df.empty: continue
            
            sh = pre_df["High"].values[find_swing_highs(pre_df["High"].values, n=1)][-1:]
            sl = pre_df["Low"].values[find_swing_lows(pre_df["Low"].values, n=1)][-1:]
            
            win_df = day_df.iloc[win_idx:]
            # Bearish Setup (Sweep High)
            if len(sh) > 0:
                for wi, (ts, row) in enumerate(win_df.iterrows()):
                    if row["High"] > sh[0]:
                        fvgs = find_bearish_fvg(win_df.iloc[wi:])
                        if fvgs:
                            fi, g_bot, g_top = fvgs[0]
                            risk = sh[0]*(1+sl_buffer) - g_bot
                            if risk > 0:
                                setups.append({"window_name": win["name"], "window_short": win["short"], "window_color": win["color"], "date": d, "direction": "bearish", "sweep_time": ts, "sweep_price": sh[0], "fvg_top": g_top, "fvg_bottom": g_bot, "entry_price": g_bot, "sl_price": sh[0]*(1+sl_buffer), "tp_price": g_bot - DEFAULT_RR*risk, "risk_pts": risk, "bar_idx_entry": df.index.get_loc(win_df.index[wi+fi])})
                        break
            # Bullish Setup (Sweep Low)
            if len(sl) > 0:
                for wi, (ts, row) in enumerate(win_df.iterrows()):
                    if row["Low"] < sl[0]:
                        fvgs = find_bullish_fvg(win_df.iloc[wi:])
                        if fvgs:
                            fi, g_bot, g_top = fvgs[0]
                            risk = g_top - sl[0]*(1-sl_buffer)
                            if risk > 0:
                                setups.append({"window_name": win["name"], "window_short": win["short"], "window_color": win["color"], "date": d, "direction": "bullish", "sweep_time": ts, "sweep_price": sl[0], "fvg_top": g_top, "fvg_bottom": g_bot, "entry_price": g_top, "sl_price": sl[0]*(1-sl_buffer), "tp_price": g_top + DEFAULT_RR*risk, "risk_pts": risk, "bar_idx_entry": df.index.get_loc(win_df.index[wi+fi])})
                        break
    return setups

def simulate_trades(df, setups, rr, initial_cap=INITIAL_CAPITAL, risk_pct=RISK_PER_TRADE_PCT):
    if not setups: return pd.DataFrame()
    equity, records = initial_cap, []
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    for s in setups:
        bi, entry, sl, tp, risk = int(s["bar_idx_entry"]), s["entry_price"], s["sl_price"], s["tp_price"], s["risk_pts"]
        size = (equity * risk_pct / 100) / risk
        outcome, pnl_pts = "timeout", 0.0
        for j in range(bi + 1, min(bi + 25, len(df))):
            if s["direction"] == "bullish":
                if h[j] >= tp: outcome, pnl_pts = "win", tp - entry; break
                if l[j] <= sl: outcome, pnl_pts = "loss", sl - entry; break
            else:
                if l[j] <= tp: outcome, pnl_pts = "win", entry - tp; break
                if h[j] >= sl: outcome, pnl_pts = "loss", entry - sl; break
        pnl_dollar = pnl_pts * size
        equity += pnl_dollar
        records.append({**s, "outcome": outcome, "pnl_dollar": pnl_dollar, "equity_after": equity})
    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────

def main():
    # FIXED: Declare global at the very beginning of the function
    global DEFAULT_RR
    
    parser = argparse.ArgumentParser(description="ICT Silver Bullet Strategy Backtester")
    parser.add_argument("--ticker", type=str,   default=DEFAULT_TICKER)
    parser.add_argument("--days",   type=int,   default=DEFAULT_DAYS)
    parser.add_argument("--rr",     type=float, default=DEFAULT_RR)
    parser.add_argument("--swing-n", type=int,  default=DEFAULT_SWING_N)
    args = parser.parse_args()

    # Now assign the new value
    DEFAULT_RR = args.rr

    print(f"--- Running Silver Bullet Backtest: {args.ticker} ---")
    
    try:
        df = fetch_5min(args.ticker, args.days)
        setups = detect_setups(df, swing_n=args.swing_n)
        if not setups:
            print("No setups found.")
            return
        
        trades = simulate_trades(df, setups, rr=args.rr)
        
        # Simple Summary Output
        win_rate = (len(trades[trades['outcome']=='win']) / len(trades)) * 100
        print(f"Trades: {len(trades)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Final Equity: ${trades['equity_after'].iloc[-1]:,.2f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()