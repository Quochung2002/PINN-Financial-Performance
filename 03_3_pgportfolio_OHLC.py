# ============================================================================
#  Multi‑Asset Portfolio Management ― PPO • DDPG • TD3 • SAC + pgportfolio BL
# ============================================================================

import os
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# --- Data Source Import ---
import yfinance as yf # <-- Single, consistent data source

# ---------- monkey‑patch for NumPy 2.x (pgportfolio uses np.alltrue) ----------
if not hasattr(np, "alltrue"):
    np.alltrue = np.all

# ---------- SB3 imports ------------------------------------------------------
from stable_baselines3 import PPO, TD3, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines.ddpg_pinn.ddpg_pinn import DDPG_PINN
from stable_baselines.td3_pinn.td3_pinn import TD3_PINN
from stable_baselines.ppo_pinn.ppo_pinn import PPO_PINN
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
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

class BEST(TDAgent):
    """
    Best Stock Strategy - Oracle Agent.

    This agent implements the "Buy and Hold" strategy for the single best asset.
    It determines the best-performing asset by looking at the entire price
    history in advance, using the cumulative product of returns, exactly as
    specified in the original pgportfolio `BEST` agent logic.
    """
    # =======================================================================
    # THE FIX IS HERE: The __init__ method now accepts `price_history`.
    # =======================================================================
    def __init__(self, price_history: np.ndarray):
        super().__init__()
        # The agent "cheats" by receiving the full price history at initialization
        self._price_history = price_history
        self.last_b = None # This will store the static portfolio weights

    def decide_by_history(self, *args, **kwargs):
        # The core logic runs only ONCE on the first call.
        if self.last_b is None:
            # 1. Calculate daily price relatives (price_t / price_{t-1})
            price_relatives = self._price_history[1:] / self._price_history[:-1]
            price_relatives[np.isnan(price_relatives)] = 1.0 # Handle any NaNs

            # 2. Calculate the cumulative product of returns (total growth factor).
            tmp_cumprod_ret = np.cumprod(price_relatives, axis=0)

            # 3. Find the index of the asset with the highest final return.
            best_ind = np.argmax(tmp_cumprod_ret[-1, :])

            # 4. Create the static, one-hot encoded portfolio vector.
            n_assets = self._price_history.shape[1]
            self.last_b = np.zeros(n_assets)
            self.last_b[best_ind] = 1.0

        # On every call, return the same "buy-and-hold" decision vector.
        return self.last_b.ravel()
# ----------------------------------------------------------------------------
#                            DATA DOWNLOADER
# ----------------------------------------------------------------------------
def download_data(assets: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Downloads and formats data from Yahoo Finance."""
    print(f"Downloading data for {len(assets)} assets from Yahoo Finance...")
    df_raw = yf.download(assets, start=start_date, end=end_date, progress=False)
    if df_raw.empty:
        raise ValueError(f"No data downloaded for assets: {assets}. Check tickers and date range.")

    # Select only the necessary columns before processing
    df_raw = df_raw.loc[:, pd.IndexSlice[['Open', 'High', 'Low', 'Close'], :]]

    # Flatten the multi-level column index and format names
    df_raw.columns = df_raw.columns.to_flat_index()
    df = df_raw.rename(columns={col: f"{col[0].lower()}_{col[1]}" for col in df_raw.columns})

    # Forward-fill and back-fill any missing values
    df = df.ffill().bfill()
    print("Yahoo Finance data download and formatting complete.")
    return df

# ----------------------------------------------------------------------------
#                            RL ENVIRONMENT
# ----------------------------------------------------------------------------
class MultiAssetPortfolioEnv(gym.Env):
    """Long‑only, fully‑invested multi‑asset portfolio environment."""

    def __init__(self, df: pd.DataFrame, assets: list[str], window_size: int = 5):
        super().__init__()
        self.df = df
        self.assets = assets
        self.n_assets = len(assets)
        self.window_size = window_size
        self.pointer = self.window_size
        self.n_features = 4  # open, high, low, close

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

    def _get_asset_window(self, asset: str, col: str):
        series = self.df[f"{col}_{asset}"].values[
                 self.pointer - self.window_size: self.pointer
                 ]
        return np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_observation(self):
        obs = [
            self._get_asset_window(asset, col)
            for col in ("open", "high", "low", "close")
            for asset in self.assets
        ]
        return np.nan_to_num(np.array(obs).T, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pointer = self.window_size
        self.weights = np.ones(self.n_assets) / self.n_assets
        return self._get_observation(), {}

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        action = action / action.sum() if action.sum() > 0 else np.ones_like(action) / self.n_assets
        self.weights = action.copy()

        cur_prices = np.array(
            [self.df[f"close_{a}"].values[self.pointer - 1] for a in self.assets]
        )
        nxt_prices = np.array(
            [self.df[f"close_{a}"].values[self.pointer] for a in self.assets]
        )
        cur_prices = np.where(cur_prices == 0.0, 1e-6, cur_prices)
        price_relatives = (nxt_prices / cur_prices)

        reward = float(np.dot(action, price_relatives - 1.0))

        self.pointer += 1
        terminated = self.pointer >= len(self.df)
        truncated = False
        return self._get_observation(), reward, terminated, truncated, {}


# ----------------------------------------------------------------------------
#                        SIMULATION & METRICS
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


def simulate_pg_agent(agent_cls, price_data: np.ndarray, **kwargs):
    n_steps, n_assets = price_data.shape
    agent = agent_cls(**kwargs)
    wealth = [1000.0]
    last_b = np.ones(n_assets) / n_assets

    for t in range(1, n_steps):
        cur = price_data[t - 1]
        nxt = price_data[t]
        cur[cur == 0] = 1e-10
        rel = nxt / cur

        # new_b = agent.decide_by_history(rel.reshape(1, -1), last_b)
        new_b = agent.decide_by_history(rel, last_b)
        if new_b.shape != (n_assets,):
            new_b = np.ones(n_assets) / n_assets

        wealth.append(wealth[-1] * np.dot(new_b, rel))
        last_b = new_b
    return wealth


def compute_metrics(wealth, *, freq=252, risk_free_rate=0.0):
    wealth = np.asarray(wealth, dtype=float)
    if wealth.size < 2: return {"APV": np.nan, "Ann.Return(%)": np.nan, "Ann.Vol(%)": np.nan, "Sharpe": np.nan, "MDD(%)": np.nan, "Calmar": np.nan}
    rets = np.diff(wealth) / wealth[:-1]
    if rets.size == 0: return {"APV": wealth[-1], "Ann.Return(%)": 0, "Ann.Vol(%)": 0, "Sharpe": np.nan, "MDD(%)": 0, "Calmar": np.nan}
    ann_return = (wealth[-1] / wealth[0]) ** (freq / rets.size) - 1.0
    ann_vol = np.std(rets, ddof=1) * np.sqrt(freq)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
    running_max = np.maximum.accumulate(wealth)
    drawdown = (wealth - running_max) / running_max
    mdd = drawdown.min()
    calmar = ann_return / abs(mdd) if mdd != 0 else np.nan
    return {"APV": wealth[-1], "Ann.Return(%)": ann_return * 100, "Ann.Vol(%)": ann_vol * 100, "Sharpe": sharpe, "MDD(%)": mdd * 100, "Calmar": calmar}


# ============================================================================
#                         MAIN ANALYSIS PIPELINE
# ============================================================================
def run_portfolio_analysis(portfolio_spec: dict):
    """Runs the full pipeline for a given portfolio specification."""
    name = portfolio_spec['name']
    assets = portfolio_spec['assets']
    start_time = time.time()
    print(f"\n{'='*80}\nRunning Analysis for Portfolio: {name}\n{'='*80}")
    

    # --- Download, Split, and Verify Data ---
    try:
        full_df = download_data(assets, start_date=portfolio_spec['train_start'], end_date=portfolio_spec['test_end'])
        train_df = full_df.loc[portfolio_spec['train_start']:portfolio_spec['train_end']].reset_index(drop=True)
        test_df = full_df.loc[portfolio_spec['test_start']:portfolio_spec['test_end']].reset_index(drop=True)

        if train_df.empty or test_df.empty or train_df.isnull().values.any() or test_df.isnull().values.any():
            print(f"Data for {name} is missing or incomplete for the specified dates. Skipping analysis.")
            return
    except Exception as e:
        print(f"Could not run analysis for {name}. Reason: {e}")
        return

    window_size = 5
    SB3_ALGOS = {"PPO_PINN":PPO_PINN,"TD3_PINN":TD3_PINN,"DDPG_PINN":DDPG_PINN,"PPO": PPO, "DDPG": DDPG, "TD3": TD3, "SAC": SAC}
    #SB3_ALGOS = {"PPO_PINN":PPO_PINN}
    #SB3_ALGOS={}
    #SB3_ALGOS = {"DDPG_PINN": DDPG_PINN}
    results = {}
    metrics_rows = []

    # --- Train and Evaluate RL Models ---
    for algo_name, Algo in SB3_ALGOS.items():
        print(f"\n=== Training {algo_name} on {name} ===")
        train_env = DummyVecEnv([lambda: Monitor(MultiAssetPortfolioEnv(train_df, assets, window_size))])
        if algo_name in ["PPO_PINN","TD3_PINN"]:
            # Custom PINN models may have different policy aliases or defaults
            model = Algo(env=train_env, seed=SEED, verbose=0)
        else:
            # Standard SB3 models
            model = Algo(policy="MlpPolicy", env=train_env, seed=SEED, verbose=0)
        model.learn(total_timesteps=20_000)
        print(f"{algo_name} training done.")

        eval_env = MultiAssetPortfolioEnv(test_df, assets, window_size)
        wealth = simulate_sb3_strategy(eval_env, model)
        results[algo_name] = wealth
        metrics_rows.append({"Algorithm": algo_name, **compute_metrics(wealth)})

    # --- Simulate Benchmark Agents ---
    test_close = np.vstack([test_df[f"close_{a}"].values for a in assets]).T
    pg_agents = {"UBAH": (UBAH, {}), "CRP": (CRP, {}), "OLMAR": (OLMAR, {"window": 5, "eps": 10}), "RMR": (RMR, {"eps": 10}), "PAMR": (PAMR, {"eps": 10, "C": 0.5})}#,"BEST": (BEST, {"price_history": test_close}) }
    for label, (cls, params) in pg_agents.items():
        print(f"Simulating {label} on {name}…")
        wealth = simulate_pg_agent(cls, test_close, **params)
        results[label] = wealth
        metrics_rows.append({"Algorithm": label, **compute_metrics(wealth)})
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time for {name}: {elapsed_time:.2f} seconds")

    # --- Display Results ---
    metrics_df = pd.DataFrame(metrics_rows).set_index("Algorithm").reindex(["PPO_PINN","TD3_PINN","DDPG_PINN","PPO", "DDPG", "TD3", "SAC", "UBAH", "CRP", "OLMAR", "RMR", "PAMR"]).round(2)
    # metrics_df = pd.DataFrame(metrics_rows).set_index("Algorithm").reindex(
    #     ["DDPG_PINN", "UBAH", "CRP", "OLMAR", "RMR", "PAMR", "BEST"]).round(2)
    print(f"\n----------------  Performance Comparison: {name}  ----------------")
    print(metrics_df.to_string())

    # --- Plot Results ---
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
    # Note: Vietnamese tickers now have a .VN suffix for use with yfinance
    PORTFOLIOS = [
        {
            'name': 'S&P 100 (U.S.)',
            'assets': ['NVDA', 'AAPL', 'GOOGL', 'AMZN', 'T', 'BAC', 'PFE', 'UNH', 'PYPL', 'KO'],
            'train_start': '2015-01-01', 'train_end': '2023-01-01',
            'test_start': '2023-01-02', 'test_end': '2025-01-01'
        },

        {
            'name': 'VN100 (Vietnam)',
            'assets': ['HPG.VN', 'VIX.VN', 'SSI.VN', 'DIG.VN', 'MSN.VN', 'STB.VN', 'VNM.VN', 'HSG.VN', 'FPT.VN', 'DPM.VN'],
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