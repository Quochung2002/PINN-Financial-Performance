# =============================================================================
#  DRL Portfolio: Full Algo Set (PPO, DDPG, TD3, SAC + PINN) with Fair Costs
#  - State = windowed tech features + previous weights
#  - Reward = log-net growth (stable) with L1 turnover commission (0.25%)
#  - P0 = 10,000 USD
#  - Tech-only features; z-scored by train stats; VecNormalize on top
#  - Baselines (pgportfolio) pay the same commission (except UBAH which never rebalances)
# =============================================================================

import os
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# Data
import yfinance as yf

# SB3
from stable_baselines3 import PPO, DDPG, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Optional custom PINN algos (you had these in your original code)
from stable_baselines.ddpg_pinn.ddpg_pinn import DDPG_PINN
from stable_baselines.td3_pinn.td3_pinn import TD3_PINN
from stable_baselines.ppo_pinn.ppo_pinn import PPO_PINN

# Baselines (pgportfolio)
from pgportfolio.tdagent.tdagent import TDAgent
from pgportfolio.tdagent.algorithms.olmar import OLMAR
from pgportfolio.tdagent.algorithms.rmr import RMR
from pgportfolio.tdagent.algorithms.pamr import PAMR
from pgportfolio.tdagent.algorithms.crp import CRP
from pgportfolio.tdagent.algorithms.ubah import UBAH

# Repro
import random
import torch

# ---------- suppress TF clutter (pgportfolio uses TF internally) -------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)

# ---------- NumPy 2.x monkey patch for pgportfolio ---------------------------
if not hasattr(np, "alltrue"):
    np.alltrue = np.all

# ---------- Seeding ----------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
#                                DATA
# =============================================================================
def download_data(assets: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Downloading data for {len(assets)} assets from Yahoo Finance...")
    df_raw = yf.download(assets, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if df_raw.empty:
        raise ValueError(f"No data downloaded for assets: {assets}. Check tickers and dates.")
    keep = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df_raw = df_raw.loc[:, pd.IndexSlice[keep, :]]
    df_raw.columns = df_raw.columns.to_flat_index()
    df = df_raw.rename(columns={col: f"{col[0].lower().replace(' ', '_')}_{col[1]}" for col in df_raw.columns})
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    print("Data download complete.")
    return df


# =============================================================================
#                        FEATURE ENGINEERING (TECH ONLY)
# =============================================================================
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

def _bb_z(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    ma = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    z = (close - ma) / (k * sd).replace(0, np.nan)
    return z.clip(-3, 3).fillna(0.0)

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

def engineer_features(df: pd.DataFrame, assets: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    for a in assets:
        c = out[f"close_{a}"]
        h = out[f"high_{a}"]
        l = out[f"low_{a}"]
        v = out[f"volume_{a}"].replace(0, np.nan)

        # Returns & vol
        out[f"ret1_{a}"]   = c.pct_change().fillna(0.0).clip(-0.2, 0.2)
        out[f"ret5_{a}"]   = c.pct_change(5).fillna(0.0).clip(-0.5, 0.5)
        out[f"rvol20_{a}"] = c.pct_change().rolling(20, min_periods=20).std().fillna(method="bfill")
        out[f"mom10_{a}"]  = (c / c.shift(10) - 1.0).fillna(0.0).clip(-1, 1)

        # Oscillators & trend
        out[f"rsi14_{a}"]  = _rsi(c, 14) / 100.0
        macd     = _ema(c, 12) - _ema(c, 26)
        macd_sig = _ema(macd, 9)
        out[f"macd_{a}"]   = macd.fillna(0.0)
        out[f"macds_{a}"]  = macd_sig.fillna(0.0)
        out[f"bbz20_{a}"]  = _bb_z(c, 20, 2.0)

        out[f"atr14_{a}"]  = _atr(h, l, c, 14)
        out[f"adx14_{a}"]  = (_adx(h, l, c, 14) / 100.0).fillna(0.2)

        # Volume-based
        lv = np.log(v)
        out[f"volz60_{a}"] = _roll_z(lv, 60)
        out[f"obv_{a}"]    = _obv(c, v)

        # Normalize ATR by price
        out[f"atr14_{a}"]  = (out[f"atr14_{a}"] / c.replace(0, np.nan)).fillna(0.0).clip(0, 0.2)

    # Clean & trim warmup
    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    out = out.iloc[60:].copy()

    # TECH-ONLY FEATURE LIST (drop OHLCV levels)
    FEATURES = ["ret1", "ret5", "rvol20", "mom10",
                "rsi14", "macd", "macds", "bbz20",
                "atr14", "adx14", "volz60", "obv"]
    return out, FEATURES


def standardize_split(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      assets: list[str], features: list[str]):
    cols = [f"{feat}_{a}" for feat in features for a in assets]
    mu = train_df[cols].mean()
    sigma = train_df[cols].std(ddof=0).replace(0.0, 1.0)
    train_df[cols] = (train_df[cols] - mu) / sigma
    test_df[cols]  = (test_df[cols]  - mu) / sigma
    return train_df, test_df


# =============================================================================
#                                ENV
# =============================================================================
class MultiAssetPortfolioEnv(gym.Env):
    """Long-only, fully-invested portfolio env with prev-weights in state and L1 commission costs."""

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, assets: list[str], features: list[str],
                 window_size: int = 20,
                 commission_rate: float = 0.0025,   # 0.25% per unit turnover
                 initial_capital: float = 10_000.0, # P0
                 reward_type: str = "log_net"):     # {"log_net","linear_net"}
        super().__init__()
        self.df = df
        self.assets = assets
        self.feature_list = features
        self.n_assets = len(assets)
        self.window_size = window_size
        self.pointer = self.window_size
        self.n_features = len(self.feature_list)

        self.commission_rate = float(commission_rate)
        self.initial_capital = float(initial_capital)
        self.reward_type = reward_type

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        obs_dim = self.n_assets * self.n_features + self.n_assets  # + prev weights
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.window_size, obs_dim),
                                            dtype=np.float32)

        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = self.initial_capital

    def _get_window(self, asset: str, feat: str):
        series = self.df[f"{feat}_{asset}"].values[self.pointer - self.window_size: self.pointer]
        return np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_observation(self):
        core = [
            self._get_window(asset, feat)
            for feat in self.feature_list
            for asset in self.assets
        ]
        core = np.array(core).T  # (W, N*F)
        wmat = np.repeat(self.weights.reshape(1, -1), self.window_size, axis=0)  # (W, N)
        obs = np.concatenate([core, wmat], axis=1)  # (W, N*F + N)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _price_relatives(self):
        cur = np.array([self.df[f"close_{a}"].values[self.pointer - 1] for a in self.assets])
        nxt = np.array([self.df[f"close_{a}"].values[self.pointer]     for a in self.assets])
        cur = np.where(cur == 0.0, 1e-6, cur)
        rel = nxt / cur
        return np.nan_to_num(rel, nan=1.0, posinf=1.0, neginf=1.0)

    def _turnover_mu(self, w_from: np.ndarray, w_to: np.ndarray) -> float:
        turnover = float(np.abs(w_to - w_from).sum())
        mu = 1.0 - self.commission_rate * turnover
        return max(mu, 0.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pointer = self.window_size
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = self.initial_capital
        return self._get_observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=float).ravel()
        action = np.clip(action, 0.0, 1.0)
        action = action / action.sum() if action.sum() > 0 else np.ones_like(action) / self.n_assets

        y = self._price_relatives()                   # price relatives y_t
        gross_growth = float(np.dot(self.weights, y)) # G_t = w_{t-1}^T y_t

        # Drift then rebalance
        w_drift = self.weights * y
        w_drift = w_drift / w_drift.sum()
        mu = self._turnover_mu(w_drift, action)

        # Update portfolio value
        net_growth = gross_growth * mu
        self.portfolio_value *= net_growth

        # Reward
        if self.reward_type == "log_net":
            reward = float(np.log(max(net_growth, 1e-12)))
        else:
            reward = float(net_growth - 1.0)

        # Set next weights
        self.weights = action.copy()

        # Next step
        self.pointer += 1
        terminated = self.pointer >= len(self.df)
        truncated = False
        info = {"gross_growth": gross_growth, "mu": mu, "net_growth": net_growth, "portfolio_value": self.portfolio_value}
        return self._get_observation(), reward, terminated, truncated, info


# =============================================================================
#                              BASELINES (FAIR COSTS)
# =============================================================================
def _simplex_project(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).ravel()
    w[w < 0] = 0.0
    s = w.sum()
    return (w / s) if s > 0 else np.ones_like(w) / w.size

def simulate_pg_agent(agent_cls, price_data: np.ndarray, commission_rate: float = 0.0025, **kwargs):
    """
    pgportfolio agents with same commission model.
    - UBAH: buy-and-hold (no rebalancing) -> pays no ongoing costs.
    - Others rebalance to agent's target -> pay L1-turnover commission.
    """
    n_steps, n_assets = price_data.shape
    agent = agent_cls(**kwargs)
    wealth = [10_000.0]  # P0
    w = np.ones(n_assets, dtype=float) / n_assets

    for t in range(1, n_steps):
        prev = price_data[t - 1].astype(float, copy=False)
        cur  = price_data[t].astype(float, copy=False)
        prev = np.where(prev == 0.0, 1e-10, prev)
        rel  = cur / prev
        rel  = np.nan_to_num(rel, nan=1.0, posinf=1.0, neginf=1.0)

        if agent_cls.__name__.upper() == "UBAH":
            b_t = w
            mu = 1.0
        else:
            try:
                b_t = agent.decide_by_history(rel, w)
            except TypeError:
                b_t = agent.decide_by_history(rel)
            b_t = np.asarray(b_t, dtype=float).ravel()
            if b_t.size != n_assets or not np.all(np.isfinite(b_t)):
                b_t = w
            if (b_t < 0).any() or not np.isclose(b_t.sum(), 1.0):
                b_t = _simplex_project(b_t)
            w_drift = w * rel
            w_drift = w_drift / w_drift.sum()
            turnover = np.abs(b_t - w_drift).sum()
            mu = max(1.0 - commission_rate * turnover, 0.0)

        gross = float(np.dot(w, rel))
        net = gross * mu
        wealth.append(wealth[-1] * net)

        w = b_t * rel
        w = w / w.sum()
    return wealth


# =============================================================================
#                             METRICS & EVAL
# =============================================================================
def compute_metrics(wealth, *, freq=252, risk_free_rate=0.0):
    wealth = np.asarray(wealth, dtype=float)
    if wealth.size < 2:
        return {"APV": np.nan, "Ann.Return(%)": np.nan, "Ann.Vol(%)": np.nan, "Sharpe": np.nan, "MDD(%)": np.nan, "Calmar": np.nan}
    rets = np.diff(wealth) / wealth[:-1]
    if rets.size == 0:
        return {"APV": wealth[-1], "Ann.Return(%)": 0, "Ann.Vol(%)": 0, "Sharpe": np.nan, "MDD(%)": 0, "Calmar": np.nan}
    ann_return = (wealth[-1] / wealth[0]) ** (freq / rets.size) - 1.0
    ann_vol = np.std(rets, ddof=1) * np.sqrt(freq)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
    running_max = np.maximum.accumulate(wealth)
    drawdown = (wealth - running_max) / running_max
    mdd = drawdown.min()
    calmar = ann_return / abs(mdd) if mdd != 0 else np.nan
    return {"APV": wealth[-1], "Ann.Return(%)": ann_return * 100, "Ann.Vol(%)": ann_vol * 100,
            "Sharpe": sharpe, "MDD(%)": mdd * 100, "Calmar": calmar}

def evaluate_vec(model, vec_env, steps=None, initial_capital=10_000.0, reward_type="log_net"):
    obs = vec_env.reset()
    wealth = [initial_capital]
    i = 0
    max_steps = steps or 10**9
    while i < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        r = float(rewards[0])
        if reward_type == "log_net":
            wealth.append(wealth[-1] * np.exp(r))
        else:
            wealth.append(wealth[-1] * (1.0 + r))
        if dones[0]:
            break
        i += 1
    return wealth


# =============================================================================
#                               ALGO CONFIG
# =============================================================================
def get_algo_and_kwargs(algo_name):
    import torch.nn as nn
    # Base kwargs; many algos will override some keys
    base = dict(seed=SEED, verbose=0)

    if algo_name == "PPO":
        return PPO, dict(
            learning_rate=3e-4, n_steps=4096, batch_size=256,
            gamma=0.999, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256], activation_fn=nn.Tanh), **base
        )
    if algo_name == "DDPG":
        return DDPG, dict(
            learning_rate=1e-3, buffer_size=200_000, learning_starts=1_000,
            batch_size=256, tau=0.005, gamma=0.999, train_freq=(1, "step"),
            gradient_steps=1, action_noise=None,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])), **base
        )
    if algo_name == "TD3":
        return TD3, dict(
            learning_rate=1e-3, buffer_size=200_000, learning_starts=1_000,
            batch_size=256, tau=0.005, gamma=0.999, train_freq=(1, "step"),
            gradient_steps=1, policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])), **base
        )
    if algo_name == "SAC":
        return SAC, dict(
            learning_rate=3e-4, buffer_size=300_000, learning_starts=2_000,
            batch_size=256, tau=0.005, gamma=0.999, train_freq=(1, "step"),
            gradient_steps=1, ent_coef="auto",
            policy_kwargs=dict(net_arch=[256, 256]), **base
        )
    if algo_name in ["PPO_PINN", "TD3_PINN", "DDPG_PINN"]:
        # Custom classes you already use; assume defaults are fine
        # (many custom repos have their own signatures)
        return {"PPO_PINN": PPO_PINN, "TD3_PINN": TD3_PINN, "DDPG_PINN": DDPG_PINN}[algo_name], dict(**base)

    raise ValueError(f"Unknown algo {algo_name}")


# =============================================================================
#                                 MAIN
# =============================================================================
def run_portfolio_analysis(portfolio_spec: dict,
                           train_steps_map: dict | None = None,
                           window_size: int = 20,
                           commission_rate: float = 0.0025,
                           reward_type: str = "log_net"):
    name = portfolio_spec['name']
    assets = portfolio_spec['assets']
    print(f"\n{'='*90}\nRunning Analysis for Portfolio: {name}\n{'='*90}")
    start_time = time.time()
    print(f"\n{'='*80}\nRunning Analysis for Portfolio: {name}\n{'='*80}")

    # 1) Download + Features
    try:
        full_df = download_data(assets, start_date=portfolio_spec['train_start'], end_date=portfolio_spec['test_end'])
        full_df, FEATURES = engineer_features(full_df, assets)
        # Split AFTER eng
        train_df = full_df.loc[portfolio_spec['train_start']:portfolio_spec['train_end']].reset_index(drop=True)
        test_df  = full_df.loc[portfolio_spec['test_start'] :portfolio_spec['test_end'] ].reset_index(drop=True)
        if train_df.empty or test_df.empty or train_df.isnull().values.any() or test_df.isnull().values.any():
            print(f"Data for {name} missing/incomplete—skipping.")
            return
    except Exception as e:
        print(f"Could not run analysis for {name}. Reason: {e}")
        return

    # 2) Train-only z-score
    train_df, test_df = standardize_split(train_df, test_df, assets, FEATURES)

    # 3) Algo list (your original full set)
    SB3_ALGOS = {
        "TD3_PINN": TD3_PINN, "PPO_PINN": PPO_PINN, "DDPG_PINN": DDPG_PINN,
        "PPO": PPO, "DDPG": DDPG, "TD3": TD3, "SAC": SAC
    }
    # SB3_ALGOS = {
    #     "PPO": PPO, "DDPG": DDPG, "TD3": TD3, "SAC": SAC
    # }

    # Default train steps if not provided
    if train_steps_map is None:
        train_steps_map = {
            "PPO": 100_000,
            "DDPG": 120_000,
            "TD3": 150_000,
            "SAC": 150_000,
            "PPO_PINN": 120_000,
            "TD3_PINN": 150_000,
            "DDPG_PINN": 120_000,
        }

    results = {}
    metrics_rows = []

    # --- Train & Evaluate each RL algorithm ---
    for algo_name, Algo in SB3_ALGOS.items():
        print(f"\n=== Training {algo_name} on {name} ===")

        def make_train_env():
            return Monitor(MultiAssetPortfolioEnv(
                train_df, assets, FEATURES, window_size=window_size,
                commission_rate=commission_rate,
                initial_capital=10_000.0,
                reward_type=reward_type
            ))

        def make_eval_env():
            return Monitor(MultiAssetPortfolioEnv(
                test_df, assets, FEATURES, window_size=window_size,
                commission_rate=commission_rate,
                initial_capital=10_000.0,
                reward_type=reward_type
            ))

        train_env = DummyVecEnv([make_train_env])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # Algo-specific kwargs
        AlgoClass, kwargs = get_algo_and_kwargs(algo_name) if algo_name in ["PPO","DDPG","TD3","SAC"] else (Algo, {"seed": SEED, "verbose": 0})

        # Instantiate
        if algo_name in ["PPO","DDPG","TD3","SAC", "DDPG_PINN"]:
            model = AlgoClass("MlpPolicy", env=train_env, **kwargs)
        else:
            # PINN classes in your repo usually accept just env + seed + verbose
            model = AlgoClass(env=train_env, **kwargs)

        model.learn(total_timesteps=train_steps_map.get(algo_name, 600_000))

        # Save VecNormalize stats and evaluate with same stats
        vec_path = f"vecnorm_{name.replace(' ', '_')}_{algo_name}.pkl"
        train_env.save(vec_path)

        eval_env = DummyVecEnv([make_eval_env])
        eval_env = VecNormalize.load(vec_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        wealth = evaluate_vec(model, eval_env, reward_type=reward_type)
        results[algo_name] = wealth
        metrics_rows.append({"Algorithm": algo_name, **compute_metrics(wealth)})

    # --- Baselines on test (apply same commission_rate) ---
    test_close = np.vstack([test_df[f"close_{a}"].values for a in assets]).T
    pg_agents = {
        "UBAH": (UBAH, {}),
        "CRP": (CRP, {}),
        "OLMAR": (OLMAR, {"window": 5, "eps": 10}),
        "RMR": (RMR, {"eps": 10}),
        "PAMR": (PAMR, {"eps": 10, "C": 0.5}),
    }
    for label, (cls, params) in pg_agents.items():
        print(f"Simulating baseline {label} on {name}…")
        w = simulate_pg_agent(cls, test_close, commission_rate=commission_rate, **params)
        results[label] = w
        metrics_rows.append({"Algorithm": label, **compute_metrics(w)})

    # Metrics table
    algo_order = ["PPO_PINN","TD3_PINN","DDPG_PINN","PPO","DDPG","TD3","SAC","UBAH","CRP","OLMAR","RMR","PAMR"]
    metrics_df = (
        pd.DataFrame(metrics_rows)
        .set_index("Algorithm")
        .reindex(algo_order)
        .round(2)
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n----------------  Performance Comparison: {name}  ----------------")
    print(metrics_df.to_string())
    

    # Plot
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(18, 9))
    for res_name, wealth in results.items():
        ls = "--" if res_name in {"UBAH", "CRP"} else "-"
        lw = 2.5 if res_name in SB3_ALGOS else 1.8
        plt.plot(wealth, label=res_name, linestyle=ls, linewidth=lw)
    plt.title(f"Portfolio Wealth ({name}) — Test Period", fontsize=16)
    plt.xlabel("Trading Days", fontsize=12)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.legend(ncol=4, fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    PORTFOLIOS = [
        {
            'name': 'S&P 100 (U.S.)',
            'assets': ['NVDA', 'AAPL', 'GOOGL', 'AMZN', 'T', 'BAC', 'PFE', 'UNH', 'PYPL', 'KO'],
            'train_start': '2015-01-01', 'train_end': '2023-01-01',
            'test_start':  '2023-01-02', 'test_end':  '2025-01-01'
        },
        {
            'name': 'VN100 (Vietnam)',
            'assets': ['HPG.VN', 'VIX.VN', 'SSI.VN', 'DIG.VN', 'MSN.VN', 'STB.VN', 'VNM.VN', 'HSG.VN', 'FPT.VN', 'DPM.VN'],
            'train_start': '2015-01-01', 'train_end': '2023-01-01',
            'test_start':  '2023-01-02', 'test_end':  '2025-01-01'
        },
        {
            'name': 'CSI 300 (China)',
            'assets': ['601868.SS', '600010.SS', '601669.SS', '300059.SZ', '601398.SS', '600111.SS', '603993.SS', '600050.SS', '600515.SS', '600875.SS'],
            'train_start': '2015-01-01', 'train_end': '2023-01-01',
            'test_start':  '2023-01-02', 'test_end':  '2025-01-01'
        }
    ]

    # You can shorten steps for a smoke test by editing train_steps_map per algo
    for portfolio in PORTFOLIOS:
        run_portfolio_analysis(
            portfolio_spec=portfolio,
            train_steps_map={
            "PPO": 100_000,
            "DDPG": 120_000,
            "TD3": 150_000,
            "SAC": 150_000,
            "PPO_PINN": 120_000,
            "TD3_PINN": 150_000,
            "DDPG_PINN": 120_000,
            },
            window_size=5,
            commission_rate=0.0025,  # 0.25% turnover commission
            reward_type="log_net"    # or "linear_net"
        )
