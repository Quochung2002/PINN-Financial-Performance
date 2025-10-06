# ============================================================================
#  Multi-Asset Portfolio Management ― PPO • DDPG • TD3 • SAC + pgportfolio BL
#  (Upgraded input: OHLCV + technical indicators; training logic unchanged)
# ============================================================================

import os
import logging
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# --- Data Source Import ---
import yfinance as yf  # <-- Single, consistent data source

# ---------- monkey-patch for NumPy 2.x (pgportfolio uses np.alltrue) ----------
if not hasattr(np, "alltrue"):
    np.alltrue = np.all

# ---------- SB3 imports ------------------------------------------------------
from stable_baselines3 import PPO, TD3, SAC, DDPG, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines.ddpg_pinn.ddpg_pinn import DDPG_PINN
from stable_baselines.td3_pinn.td3_pinn import TD3_PINN
from stable_baselines.ppo_pinn.ppo_pinn import PPO_PINN
from stable_baselines.a2c_pinn.a2c_pinn import A2C_PINN

# ---------- suppress TensorFlow clutter (pgportfolio) ------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)

# ---------- pgportfolio online agents ---------------------------------------
from pgportfolio.tdagent.tdagent import TDAgent
from pgportfolio.tdagent.algorithms.olmar import OLMAR
from pgportfolio.tdagent.algorithms.rmr import RMR
from pgportfolio.tdagent.algorithms.pamr import PAMR
from pgportfolio.tdagent.algorithms.crp import CRP
from pgportfolio.tdagent.algorithms.ubah import UBAH
# from pgportfolio.tdagent.algorithms.best import BEST
import random
import torch

# --- Set a seed for consistent results ---
COMMISSION_RATE = 0.0025   # 0.25% per-dollar traded
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

class BEST(TDAgent):
    """
    Best Stock Strategy - Oracle Agent.
    """
    def __init__(self, price_history: np.ndarray):
        super().__init__()
        self._price_history = price_history
        self.last_b = None

    def decide_by_history(self, *args, **kwargs):
        if self.last_b is None:
            price_relatives = self._price_history[1:] / self._price_history[:-1]
            price_relatives[np.isnan(price_relatives)] = 1.0
            tmp_cumprod_ret = np.cumprod(price_relatives, axis=0)
            best_ind = np.argmax(tmp_cumprod_ret[-1, :])
            n_assets = self._price_history.shape[1]
            self.last_b = np.zeros(n_assets)
            self.last_b[best_ind] = 1.0
        return self.last_b.ravel()

# ----------------------------------------------------------------------------
#                            DATA DOWNLOADER (OHLCV)
# ----------------------------------------------------------------------------
def download_data(assets: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Downloads OHLCV (+ Adj Close) from Yahoo Finance and flattens columns."""
    print(f"Downloading data for {len(assets)} assets from Yahoo Finance...")
    df_raw = yf.download(assets, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if df_raw.empty:
        raise ValueError(f"No data downloaded for assets: {assets}. Check tickers and date range.")

    # Keep OHLCV + Adj Close
    keep = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df_raw = df_raw.loc[:, pd.IndexSlice[keep, :]]

    # Flatten MultiIndex -> feature_asset (lowercase, space->underscore)
    df_raw.columns = df_raw.columns.to_flat_index()
    df = df_raw.rename(columns={col: f"{col[0].lower().replace(' ', '_')}_{col[1]}" for col in df_raw.columns})

    # Fill gaps
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    print("Yahoo Finance data download and formatting complete.")
    return df

# ----------------------------------------------------------------------------
#                     TECHNICAL INDICATORS / FEATURE ENGINEERING
# ----------------------------------------------------------------------------
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def _bbands(close: pd.Series, n: int = 20, k: float = 2.0):
    ma = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return ma, upper, lower


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean().fillna(method="bfill")

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([(high - low),
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    tr_n = tr.rolling(n, min_periods=n).sum()

    plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(n, min_periods=n).sum() / tr_n.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(n, min_periods=n).sum() / tr_n.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.rolling(n, min_periods=n).mean().fillna(method="bfill")
    return adx.clip(0, 100).fillna(20.0)

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0.0)
    obv = (sign * volume).fillna(0.0).cumsum()
    obv_pct = obv.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1, 1)
    return obv_pct

def _roll_z(s: pd.Series, n: int = 60) -> pd.Series:
    m = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return ((s - m) / sd.replace(0, np.nan)).fillna(0.0).clip(-5, 5)
def _willr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    wr = -100 * (hh - close) / (hh - ll).replace(0, np.nan)
    # Return classic %R in [-100, 0]; we’ll scale to [0,1] at feature time.
    return wr.clip(-100, 0)

def engineer_features(df: pd.DataFrame, assets: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Create per-asset features named <feature>_<ASSET>.
    Returns (features_df, feature_list_in_order)
    """
    out = df.copy()

    for a in assets:
        c = out[f"close_{a}"]
        o = out[f"open_{a}"]
        h = out[f"high_{a}"]
        l = out[f"low_{a}"]
        v = out[f"volume_{a}"].replace(0, np.nan)

        # Price/return basics
        out[f"ret1_{a}"]   = c.pct_change().fillna(0.0).clip(-0.2, 0.2)
        out[f"ret5_{a}"]   = c.pct_change(5).fillna(0.0).clip(-0.5, 0.5)
        out[f"rvol20_{a}"] = c.pct_change().rolling(20, min_periods=20).std().fillna(method="bfill")
        out[f"mom10_{a}"]  = (c / c.shift(10) - 1.0).fillna(0.0).clip(-1, 1)

        # RSI / MACD / Bollinger-z
        out[f"rsi14_{a}"]  = _rsi(c, 14) / 100.0
        macd     = _ema(c, 12) - _ema(c, 26)
        macd_sig = _ema(macd, 9)
        out[f"macd_{a}"]   = macd.fillna(0.0)
        out[f"macds_{a}"]  = macd_sig.fillna(0.0)
        bb_mid, bb_up, bb_low = _bbands(c, 20, 2.0)
        # %B in [0,1] (clip for safety)
        out[f"bbp20_{a}"] = ((c - bb_low) / (bb_up - bb_low).replace(0, np.nan)).clip(0, 1)
        # Bandwidth (relative width); safe when ma>0
        out[f"bbw20_{a}"] = ((bb_up - bb_low) / bb_mid.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(
            0.0)

        # Volatility / trend strength
        out[f"atr14_{a}"]  = _atr(h, l, c, 14)
        out[f"adx14_{a}"]  = (_adx(h, l, c, 14) / 100.0).fillna(0.2)

        # Volume-derived
        lv = np.log(v)

        out[f"obv_{a}"]    = _obv(c, v)
        # Williams’ %R (14) — scaled to [0,1] for NN stability
        # classic %R is [-100,0]; map to [0,1] via (wr + 100)/100
        out[f"willr14_{a}"] = (_willr(h, l, c, 14) + 100.0) / 100.0

        # Normalize ATR by price for stability
        out[f"atr14_{a}"]  = (out[f"atr14_{a}"] / c.replace(0, np.nan)).fillna(0.0).clip(0, 0.2)

    # Feature order for stacking
    base = ["open", "high", "low", "close", "volume"]
    tech = ["ret1", "ret5", "rvol20", "mom10", "rsi14", "macd", "macds", "bbp20", "atr14", "adx14", "willr14", "obv"]
    FEATURES = base + tech

    # Clean and trim warmup
    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    out = out.iloc[60:].copy()  # drop early rows with heavy rolling NaNs

    return out, FEATURES
# ----------------------------------------------------------------------------
#                            RL ENVIRONMENT (features-ready)
# ----------------------------------------------------------------------------
class MultiAssetPortfolioEnv(gym.Env):
    """Long-only, fully-invested multi-asset portfolio environment with arbitrary feature set."""

    def __init__(self, df: pd.DataFrame, assets: list[str], features: list[str], window_size: int = 5):
        super().__init__()
        self.df = df
        self.assets = assets
        self.feature_list = features
        self.n_assets = len(assets)
        self.window_size = window_size
        self.pointer = self.window_size
        self.n_features = len(self.feature_list)

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.n_assets * self.n_features),
            dtype=np.float32,
        )
        self.weights = np.ones(self.n_assets) / self.n_assets

    def _get_window(self, asset: str, feat: str):
        series = self.df[f"{feat}_{asset}"].values[self.pointer - self.window_size: self.pointer]
        return np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_observation(self):
        obs = [
            self._get_window(asset, feat)
            for feat in self.feature_list
            for asset in self.assets
        ]
        return np.nan_to_num(np.array(obs).T, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pointer = self.window_size
        self.weights = np.ones(self.n_assets) / self.n_assets
        return self._get_observation(), {}

    def step(self, action):
        # normalize action (long-only, fully invested)
        action = np.clip(action, 0.0, 1.0)
        action = action / action.sum() if action.sum() > 0 else np.ones_like(action) / self.n_assets

        # --- prices & relatives
        cur_prices = np.array([self.df[f"close_{a}"].values[self.pointer - 1] for a in self.assets])
        nxt_prices = np.array([self.df[f"close_{a}"].values[self.pointer] for a in self.assets])
        cur_prices = np.where(cur_prices == 0.0, 1e-6, cur_prices)
        rel = nxt_prices / cur_prices  # price relatives for this step

        # --- commission: turnover between previous weights and new action
        turnover = float(np.sum(np.abs(action - self.weights)))
        trade_cost = COMMISSION_RATE * turnover

        # --- portfolio growth (apply rebalanced weights to next-step relatives)
        gross_growth = float(np.dot(action, rel))
        net_growth = (1.0 - trade_cost) * gross_growth

        # use multiplicative reward: growth - 1.0 (so wealth *= 1+reward)
        reward = net_growth - 1.0

        # commit new weights AFTER rebalancing
        self.weights = action.copy()

        self.pointer += 1
        terminated = self.pointer >= len(self.df)
        truncated = False
        return self._get_observation(), reward, terminated, truncated, {}


# ----------------------------------------------------------------------------
#                        SIMULATION & METRICS (unchanged)
# ----------------------------------------------------------------------------
def simulate_sb3_strategy(env: MultiAssetPortfolioEnv, model):
    obs, _ = env.reset()
    wealth = [1000.0]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        wealth.append(wealth[-1] * (1 + reward))
    return wealth

def _simplex_project(w: np.ndarray) -> np.ndarray:
    """Project arbitrary weights onto the probability simplex (nonneg, sum=1)."""
    w = np.asarray(w, dtype=float).ravel()
    w[w < 0] = 0.0
    s = w.sum()
    return (w / s) if s > 0 else np.ones_like(w) / w.size


def simulate_pg_agent(agent_cls, price_data: np.ndarray, **kwargs):
    n_steps, n_assets = price_data.shape
    agent = agent_cls(**kwargs)
    wealth = [1000.0]

    # Start equal weights
    w = np.ones(n_assets, dtype=float) / n_assets

    for t in range(1, n_steps):
        prev = price_data[t - 1].astype(float, copy=False)
        cur  = price_data[t].astype(float, copy=False)
        prev = np.where(prev == 0.0, 1e-10, prev)
        rel  = cur / prev
        rel  = np.nan_to_num(rel, nan=1.0, posinf=1.0, neginf=1.0)

        # 1) apply market move (drift)
        pre_rebalance_growth = float(np.dot(w, rel))

        # holdings drift to w_drift
        w_drift = w * rel
        s = w_drift.sum()
        w_drift = (w_drift / s) if s > 0 else np.ones_like(w_drift) / n_assets

        if agent_cls.__name__.upper() == "UBAH":
            # Buy-and-hold: no rebalancing, no commission
            b_t = w_drift
            turnover = 0.0
        else:
            try:
                b_t = agent.decide_by_history(rel, w_drift)
            except TypeError:
                b_t = agent.decide_by_history(rel)
            b_t = np.asarray(b_t, dtype=float).ravel()
            if b_t.size != n_assets or not np.all(np.isfinite(b_t)):
                b_t = w_drift
            if (b_t < 0).any() or not np.isclose(b_t.sum(), 1.0):
                b_t = _simplex_project(b_t)

            # turnover from drifted weights to new target
            turnover = float(np.sum(np.abs(b_t - w_drift)))

        trade_cost = COMMISSION_RATE * turnover

        # multiplicative wealth update: first market move, then pay commission to rebalance
        step_factor = pre_rebalance_growth * (1.0 - trade_cost)
        wealth.append(wealth[-1] * step_factor)

        # set end-of-step holdings for next period
        w = b_t

    return wealth

def compute_metrics(wealth, *, freq=252, risk_free_rate=0.0):
    wealth = np.asarray(wealth, dtype=float)
    if wealth.size < 2:
        return {
            "APV": np.nan, "Ann.Return(%)": np.nan, "Ann.Vol(%)": np.nan,
            "Sharpe": np.nan, "MDD(%)": np.nan, "Calmar": np.nan,
            "Cum.Return(%)": np.nan
        }
    rets = np.diff(wealth) / wealth[:-1]
    if rets.size == 0:
        return {
            "APV": wealth[-1], "Ann.Return(%)": 0, "Ann.Vol(%)": 0,
            "Sharpe": np.nan, "MDD(%)": 0, "Calmar": np.nan,
            "Cum.Return(%)": (wealth[-1] / wealth[0] - 1.0) * 100
        }

    ann_return = (wealth[-1] / wealth[0]) ** (freq / rets.size) - 1.0
    ann_vol = np.std(rets, ddof=1) * np.sqrt(freq)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
    running_max = np.maximum.accumulate(wealth)
    drawdown = (wealth - running_max) / running_max
    mdd = drawdown.min()
    calmar = ann_return / abs(mdd) if mdd != 0 else np.nan

    # <-- cumulative return over the test window
    cum_return = (wealth[-1] / wealth[0]) - 1.0

    return {
        "APV": wealth[-1],
        "Ann.Return(%)": ann_return * 100,
        "Ann.Vol(%)": ann_vol * 100,
        "Sharpe": sharpe,
        "MDD(%)": mdd * 100,
        "Calmar": calmar,
        "Cum.Return(%)": cum_return * 100,   # <-- shows up in your table
    }
    
# ============================================================================
#                         MAIN ANALYSIS PIPELINE
# ============================================================================
def run_portfolio_analysis(portfolio_spec: dict):
    """Runs the full pipeline for a given portfolio specification (training logic unchanged)."""
    name = portfolio_spec['name']
    assets = portfolio_spec['assets']
    start_time = time.time()
    print(f"\n{'='*80}\nRunning Analysis for Portfolio: {name}\n{'='*80}")

    # --- Download + Engineer Features, then Split ---
    try:
        full_df = download_data(assets, start_date=portfolio_spec['train_start'], end_date=portfolio_spec['test_end'])
        full_df, FEATURES = engineer_features(full_df, assets)

        # split AFTER feature engineering so indices/NaNs align
        train_df = full_df.loc[portfolio_spec['train_start']:portfolio_spec['train_end']].reset_index(drop=True)
        test_df  = full_df.loc[portfolio_spec['test_start'] :portfolio_spec['test_end'] ].reset_index(drop=True)

        if train_df.empty or test_df.empty or train_df.isnull().values.any() or test_df.isnull().values.any():
            print(f"Data for {name} is missing or incomplete for the specified dates. Skipping analysis.")
            return
    except Exception as e:
        print(f"Could not run analysis for {name}. Reason: {e}")
        return

    window_size = 5  # keep your original window; you can bump to 20 later if desired
    SB3_ALGOS = {"TD3_PINN":TD3_PINN,"PPO_PINN":PPO_PINN,"DDPG_PINN":DDPG_PINN,"PPO": PPO, "DDPG": DDPG, "TD3": TD3, "SAC": SAC, "A2C":A2C, "A2C_PINN":A2C_PINN}
    # SB3_ALGOS = {"TD3_PINN":TD3_PINN,"DDPG_PINN":DDPG_PINN, "DDPG": DDPG, "TD3": TD3}
    # SB3_ALGOS = {"DDPG":DDPG, "DDPG_PINN":DDPG_PINN}
    # SB3_ALGOS = {"A2C_PINN":A2C_PINN,"A2C":A2C}
    # SB3_ALGOS = {"PPO":PPO}
    # SB3_ALGOS = {}
    # per-algo training steps (single contiguous daily path, ~2k steps/episode)
    TRAIN_STEPS_MAP = {
        "PPO": 20_000, "PPO_PINN": 20_000,
        "TD3": 50_000, "TD3_PINN": 50_000,
        "SAC": 50_000,
        "DDPG": 30_000, "DDPG_PINN": 30_000,
        "A2C": 20_000, "A2C_PINN": 20_000
    }
    results = {}
    metrics_rows = []

    # --- Train and Evaluate RL Models (unchanged logic) ---
    for algo_name, Algo in SB3_ALGOS.items():
        print(f"\n=== Training {algo_name} on {name} ===")
        train_env = DummyVecEnv([lambda: Monitor(MultiAssetPortfolioEnv(train_df, assets, FEATURES, window_size))])
        if algo_name in ["PPO_PINN","TD3_PINN"]:
            model = Algo(env=train_env, seed=SEED, verbose=0)
        else:
            model = Algo(policy="MlpPolicy", env=train_env, seed=SEED, verbose=0)
        model.learn(total_timesteps=TRAIN_STEPS_MAP[algo_name])
        print(f"{algo_name} training done.")

        eval_env = MultiAssetPortfolioEnv(test_df, assets, FEATURES, window_size)
        wealth = simulate_sb3_strategy(eval_env, model)
        results[algo_name] = wealth
        metrics_rows.append({"Algorithm": algo_name, **compute_metrics(wealth)})

    # --- Simulate Benchmark Agents (unchanged: use pgportfolio classes) ---
    test_close = np.vstack([test_df[f"close_{a}"].values for a in assets]).T
    pg_agents = {
        "UBAH": (UBAH, {}),
        "CRP": (CRP, {}), "OLMAR": (OLMAR, {"window": 5, "eps": 10})
            , "RMR": (RMR, {"eps": 10}),
         "PAMR": (PAMR, {"eps": 10, "C": 0.5})}
        # ,"BEST": (BEST, {"price_history": test_close})
    for label, (cls, params) in pg_agents.items():
        print(f"Simulating {label} on {name}…")
        wealth = simulate_pg_agent(cls, test_close, **params)
        results[label] = wealth
        metrics_rows.append({"Algorithm": label, **compute_metrics(wealth)})

    # --- Display Results (unchanged) ---
    metrics_df = (
        pd.DataFrame(metrics_rows)
          .set_index("Algorithm")
          .reindex(["PPO_PINN","TD3_PINN","DDPG_PINN","PPO","DDPG","TD3","SAC","UBAH","CRP","OLMAR","RMR","PAMR","A2C","A2C_PINN"])
          .round(2)
    )
    end_time = time.time()
    print(f"\n----------------  Performance Comparison: {name}  ----------------")
    print(metrics_df.to_string())

    # --- Plot Results (unchanged) ---
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(16, 8))
    for res_name, wealth in results.items():
        ls = "--" if res_name in {"UBAH", "CRP", "BEST"} else "-"
        lw = 2.5 if res_name in SB3_ALGOS else 1.8
        plt.plot(wealth, label=res_name, linestyle=ls, linewidth=lw)
    plt.title(f"Portfolio Wealth ({name}) — Test Period", fontsize=16)
    plt.xlabel("Trading Days", fontsize=12)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.legend(ncol=3, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"portfolio_wealth_{name}.png")

if __name__ == "__main__":
    # Define the three portfolios with their specific assets and date ranges
    PORTFOLIOS = [
        {
            'name': 'S&P 100 (U.S.)',
            'assets': ['MSFT', 'INTC', 'BAC', 'DIS', 'PFE', 'LLY', 'NKE', 'VZ', 'WMT', 'AMGN'],
            'train_start': '2015-01-01', 'train_end': '2023-01-01',
            'test_start': '2023-01-02', 'test_end': '2025-01-01'
        },
        # {
        #     'name': 'FTSE 100 (U.K.)',
        #     'assets': ['AAL.L', 'BATS.L', 'GLEN.L', 'BT-A.L', 'DGE.L', 'GSK.L', 'HSBA.L', 'RIO.L',  'LLOY.L', 'NG.L'],
        #     'train_start': '2015-01-01', 'train_end': '2023-01-01',
        #     'test_start': '2023-01-02', 'test_end': '2025-01-01'
        # },
        {
            'name': 'VN100 (Vietnam)',
            'assets': ["HPG.VN", "VIX.VN", "SSI.VN", "DIG.VN", "MSN.VN", "STB.VN", "VNM.VN", "HSG.VN", "FPT.VN", "DPM.VN"],
            'train_start': '2015-01-01', 'train_end': '2023-01-01',
            'test_start': '2023-01-02', 'test_end': '2025-01-01'
        },
        {
            'name': 'CSI 300 (China)',
            'assets': ['601868.SS', '600010.SS', '601669.SS', '300059.SZ', '601398.SS', '600111.SS', '603993.SS', '600050.SS', '600515.SS', '600875.SS'],
            'train_start': '2015-01-01', 'train_end': '2023-01-01',
            'test_start': '2023-01-02', 'test_end': '2025-01-01'
        }
    ]

    # Loop through each portfolio and run the full analysis
    for portfolio in PORTFOLIOS:
        run_portfolio_analysis(portfolio)
