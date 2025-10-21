# ============================================================================
#  Multi-Asset Portfolio Management ― PPO • DDPG • TD3 • SAC + pgportfolio BL
#  (Upgraded: logging of rewards, loss-proxy, weights; saving plots & CSVs)
# ============================================================================

import os
import json
import logging
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)

from pgportfolio.tdagent.tdagent import TDAgent
from pgportfolio.tdagent.algorithms.olmar import OLMAR
from pgportfolio.tdagent.algorithms.rmr import RMR
from pgportfolio.tdagent.algorithms.pamr import PAMR
from pgportfolio.tdagent.algorithms.crp import CRP
from pgportfolio.tdagent.algorithms.ubah import UBAH

import random
import torch

# --- Globals / seeds ---
COMMISSION_RATE = 0.0025
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

### NEW: output root
OUTPUT_ROOT = "outputs"

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

class BEST(TDAgent):
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
    print(f"Downloading data for {len(assets)} assets from Yahoo Finance...")
    df_raw = yf.download(assets, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if df_raw.empty:
        raise ValueError(f"No data downloaded for assets: {assets}. Check tickers and date range.")
    keep = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df_raw = df_raw.loc[:, pd.IndexSlice[keep, :]]
    df_raw.columns = df_raw.columns.to_flat_index()
    df = df_raw.rename(columns={col: f"{col[0].lower().replace(' ', '_')}_{col[1]}" for col in df_raw.columns})
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

def _willr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    wr = -100 * (hh - close) / (hh - ll).replace(0, np.nan)
    return wr.clip(-100, 0)

def engineer_features(df: pd.DataFrame, assets: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    for a in assets:
        c = out[f"close_{a}"]
        h = out[f"high_{a}"]
        l = out[f"low_{a}"]
        v = out[f"volume_{a}"].replace(0, np.nan)

        out[f"ret1_{a}"]   = c.pct_change().fillna(0.0).clip(-0.2, 0.2)
        out[f"ret5_{a}"]   = c.pct_change(5).fillna(0.0).clip(-0.5, 0.5)
        out[f"rvol20_{a}"] = c.pct_change().rolling(20, min_periods=20).std().fillna(method="bfill")
        out[f"mom10_{a}"]  = (c / c.shift(10) - 1.0).fillna(0.0).clip(-1, 1)

        out[f"rsi14_{a}"]  = _rsi(c, 14) / 100.0
        macd     = _ema(c, 12) - _ema(c, 26)
        macd_sig = _ema(macd, 9)
        out[f"macd_{a}"]   = macd.fillna(0.0)
        out[f"macds_{a}"]  = macd_sig.fillna(0.0)
        bb_mid, bb_up, bb_low = _bbands(c, 20, 2.0)
        out[f"bbp20_{a}"] = ((c - bb_low) / (bb_up - bb_low).replace(0, np.nan)).clip(0, 1)
        out[f"bbw20_{a}"] = ((bb_up - bb_low) / bb_mid.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        out[f"atr14_{a}"]  = (_atr(h, l, c, 14) / c.replace(0, np.nan)).fillna(0.0).clip(0, 0.2)
        out[f"adx14_{a}"]  = (_adx(h, l, c, 14) / 100.0).fillna(0.2)

        out[f"obv_{a}"]    = _obv(c, v)
        out[f"willr14_{a}"] = (_willr(h, l, c, 14) + 100.0) / 100.0

    base = ["open", "high", "low", "close", "volume"]
    tech = ["ret1", "ret5", "rvol20", "mom10", "rsi14", "macd", "macds", "bbp20", "atr14", "adx14", "willr14", "obv"]
    FEATURES = base + tech
    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    out = out.iloc[60:].copy()
    return out, FEATURES

# ----------------------------------------------------------------------------
#                            RL ENVIRONMENT
# ----------------------------------------------------------------------------
class MultiAssetPortfolioEnv(gym.Env):
    """Long-only, fully-invested multi-asset portfolio environment with logging."""
    def __init__(self, df: pd.DataFrame, assets: list[str], features: list[str], window_size: int = 5):
        super().__init__()
        self.df = df
        self.assets = assets
        self.feature_list = features
        self.n_assets = len(assets)
        self.window_size = window_size
        self.pointer = self.window_size
        self.n_features = len(self.feature_list)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.window_size, self.n_assets * self.n_features),
                                            dtype=np.float32)
        self.weights = np.ones(self.n_assets) / self.n_assets

        ### NEW: logs for this rollout
        self.rewards_hist = []
        self.loss_proxy_hist = []   # 1 - net_growth = -reward
        self.weights_hist = []      # per-step post-rebalance weights

    def _get_window(self, asset: str, feat: str):
        series = self.df[f"{feat}_{asset}"].values[self.pointer - self.window_size: self.pointer]
        return np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_observation(self):
        obs = [self._get_window(asset, feat) for feat in self.feature_list for asset in self.assets]
        return np.nan_to_num(np.array(obs).T, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pointer = self.window_size
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.rewards_hist = []
        self.loss_proxy_hist = []
        self.weights_hist = [self.weights.copy()]
        return self._get_observation(), {}

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        action = action / action.sum() if action.sum() > 0 else np.ones_like(action) / self.n_assets
        cur_prices = np.array([self.df[f"close_{a}"].values[self.pointer - 1] for a in self.assets])
        nxt_prices = np.array([self.df[f"close_{a}"].values[self.pointer] for a in self.assets])
        cur_prices = np.where(cur_prices == 0.0, 1e-6, cur_prices)
        rel = nxt_prices / cur_prices

        turnover = float(np.sum(np.abs(action - self.weights)))
        trade_cost = COMMISSION_RATE * turnover

        gross_growth = float(np.dot(action, rel))
        net_growth = (1.0 - trade_cost) * gross_growth

        reward = net_growth - 1.0
        self.weights = action.copy()

        ### NEW: log per-step
        self.rewards_hist.append(reward)
        self.loss_proxy_hist.append(1.0 - net_growth)
        self.weights_hist.append(self.weights.copy())

        self.pointer += 1
        terminated = self.pointer >= len(self.df)
        truncated = False
        info = {"weights": self.weights.copy()}
        return self._get_observation(), reward, terminated, truncated, info

# ----------------------------------------------------------------------------
#                        ROLLOUT HELPERS & PLOTTING
# ----------------------------------------------------------------------------
def rollout_collect(env: MultiAssetPortfolioEnv, model, deterministic=True):
    """Run one full episode on the (train/test) env after training and collect logs."""
    obs, _ = env.reset()
    wealth = [1000.0]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        wealth.append(wealth[-1] * (1 + reward))
    logs = {
        "rewards": np.asarray(env.rewards_hist, float),
        "loss_proxy": np.asarray(env.loss_proxy_hist, float),
        "weights": np.asarray(env.weights_hist, float),  # T+1 x n_assets
        "wealth": np.asarray(wealth, float),
    }
    return logs

def plot_reward_loss(rewards, loss_proxy, out_path, title):
    plt.figure(figsize=(9,5))
    # rolling mean for smoother curve
    if rewards.size > 50:
        r = pd.Series(rewards).rolling(50).mean().values
        l = pd.Series(loss_proxy).rolling(50).mean().values
    else:
        r, l = rewards, loss_proxy
    plt.plot(r, label="Average reward")
    plt.plot(l, label="Training loss (proxy)")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_wealth(wealth_dict, out_path, title):
    # remove or change the seaborn style
    # plt.style.use("seaborn-v0_8")          # <- no grid
    plt.style.use("default")                 # <- classic Matplotlib, no grid

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, wealth in wealth_dict.items():
        ls = "--" if name in {"UBAH","CRP","BEST"} else "-"
        lw = 2.5 if name not in {"UBAH","CRP","BEST"} else 1.8
        ax.plot(wealth, label=name, linestyle=ls, linewidth=lw)

    ax.set_title(title)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(ncol=3, fontsize=9)

    ax.grid(False)   # <- make sure grid is off
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_weight_snapshots(
    weights, assets, out_path,
    steps=(10,20,30,40,50,60,70,80,90,100),
    style="bar",           # "bar" or "heatmap"
    n_cols=2,              # layout columns
):
    """
    weights: (T+1, n_assets) array of allocations in [0,1]
    steps:   labels for panels; indices are spaced across training automatically
    style:   "bar" (recommended) or "heatmap"
    """


    W = np.asarray(weights, float)
    T = W.shape[0] - 1
    if T < 2:  # nothing to plot
        return

    # pick evenly spaced indices across the horizon
    idx = np.linspace(1, T, num=len(steps), dtype=int)
    n_panels = len(idx)
    n_rows = int(np.ceil(n_panels / n_cols))

    # nicer tick labels
    xticks = np.arange(len(assets))
    xticklabels = [a.replace(".VN", " VN") for a in assets]

    # figure
    h_per_row = 2.2 if style == "bar" else 2.4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, h_per_row * n_rows), squeeze=False)
    plt.subplots_adjust(wspace=0.18, hspace=0.35)

    # global vmin/vmax for heatmap
    vmin, vmax = 0.0, 1.0

    # helper: tidy axes
    def _clean(ax):
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='both', length=0)

    # draw
    for k, ax in enumerate(axes.ravel()):
        _clean(ax)
        if k >= n_panels:
            ax.axis("off")
            continue

        t = idx[k]
        w = W[t]

        if style == "bar":
            bars = ax.bar(xticks, w)
            # annotate on top of bars
            for j, b in enumerate(bars):
                val = w[j]
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)
            ax.set_ylim(0, 1.02)
        else:  # heatmap
            im = ax.imshow(w[None, :], aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
            # annotate inside each cell
            for j in range(len(assets)):
                val = w[j]
                # choose contrasting text color
                txt_color = "white" if val > 0.45 else "black"
                ax.text(j, 0, f"{val:.2f}", ha="center", va="center", fontsize=8, color=txt_color)
            ax.set_yticks([])

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=8, rotation=45, ha="right")
        ax.set_title(f"Episode {steps[k]}", fontsize=10, pad=6)

    # single colorbar for heatmap
    if style == "heatmap":
        # create a small colorbar on the right
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axes[0, -1])
        cax = divider.append_axes("right", size="4%", pad=0.1)
        mappable = axes[0, 0].images[0]  # first heatmap
        fig.colorbar(mappable, cax=cax, ticks=[0, 0.5, 1.0])

    fig.suptitle("Portfolio Weights Across Training (snapshots)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
#               BENCHMARK SIM, METRICS, MAIN PIPELINE (with saving)
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
    w = np.asarray(w, dtype=float).ravel()
    w[w < 0] = 0.0
    s = w.sum()
    return (w / s) if s > 0 else np.ones_like(w) / w.size

def simulate_pg_agent(agent_cls, price_data: np.ndarray, **kwargs):
    n_steps, n_assets = price_data.shape
    agent = agent_cls(**kwargs)
    wealth = [1000.0]
    w = np.ones(n_assets, dtype=float) / n_assets
    for t in range(1, n_steps):
        prev = price_data[t - 1].astype(float, copy=False)
        cur  = price_data[t].astype(float, copy=False)
        prev = np.where(prev == 0.0, 1e-10, prev)
        rel  = cur / prev
        rel  = np.nan_to_num(rel, nan=1.0, posinf=1.0, neginf=1.0)
        pre_rebalance_growth = float(np.dot(w, rel))
        w_drift = w * rel
        s = w_drift.sum()
        w_drift = (w_drift / s) if s > 0 else np.ones_like(w_drift) / n_assets
        if agent_cls.__name__.upper() == "UBAH":
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
            turnover = float(np.sum(np.abs(b_t - w_drift)))
        trade_cost = COMMISSION_RATE * turnover
        step_factor = pre_rebalance_growth * (1.0 - trade_cost)
        wealth.append(wealth[-1] * step_factor)
        w = b_t
    return wealth

def compute_metrics(wealth, *, freq=252, risk_free_rate=0.0):
    wealth = np.asarray(wealth, dtype=float)
    if wealth.size < 2:
        return {"APV": np.nan, "Ann.Return(%)": np.nan, "Ann.Vol(%)": np.nan,
                "Sharpe": np.nan, "MDD(%)": np.nan, "Calmar": np.nan, "Cum.Return(%)": np.nan}
    rets = np.diff(wealth) / wealth[:-1]
    if rets.size == 0:
        return {"APV": wealth[-1], "Ann.Return(%)": 0, "Ann.Vol(%)": 0,
                "Sharpe": np.nan, "MDD(%)": 0, "Calmar": np.nan,
                "Cum.Return(%)": (wealth[-1] / wealth[0] - 1.0) * 100}
    ann_return = (wealth[-1] / wealth[0]) ** (freq / rets.size) - 1.0
    ann_vol = np.std(rets, ddof=1) * np.sqrt(freq)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
    running_max = np.maximum.accumulate(wealth)
    drawdown = (wealth - running_max) / running_max
    mdd = drawdown.min()
    calmar = ann_return / abs(mdd) if mdd != 0 else np.nan
    cum_return = (wealth[-1] / wealth[0]) - 1.0
    return {"APV": wealth[-1], "Ann.Return(%)": ann_return * 100, "Ann.Vol(%)": ann_vol * 100,
            "Sharpe": sharpe, "MDD(%)": mdd * 100, "Calmar": calmar, "Cum.Return(%)": cum_return * 100}

# ============================================================================
#                         MAIN ANALYSIS PIPELINE
# ============================================================================
def run_portfolio_analysis(portfolio_spec: dict):
    name = portfolio_spec['name']
    assets = portfolio_spec['assets']
    start_time = time.time()
    print(f"\n{'='*80}\nRunning Analysis for Portfolio: {name}\n{'='*80}")
    # per-portfolio output root
    port_root = os.path.join(OUTPUT_ROOT, name.replace(" ", "_"))
    _ensure_dir(port_root)

    try:
        full_df = download_data(assets, start_date=portfolio_spec['train_start'], end_date=portfolio_spec['test_end'])
        full_df, FEATURES = engineer_features(full_df, assets)
        train_df = full_df.loc[portfolio_spec['train_start']:portfolio_spec['train_end']].reset_index(drop=True)
        test_df  = full_df.loc[portfolio_spec['test_start'] :portfolio_spec['test_end'] ].reset_index(drop=True)
        if train_df.empty or test_df.empty or train_df.isnull().values.any() or test_df.isnull().values.any():
            print(f"Data for {name} is missing or incomplete for the specified dates. Skipping analysis.")
            return
    except Exception as e:
        print(f"Could not run analysis for {name}. Reason: {e}")
        return

    window_size = 5
    SB3_ALGOS = {"TD3_PINN":TD3_PINN,"PPO_PINN":PPO_PINN,"DDPG_PINN":DDPG_PINN,"PPO": PPO, "DDPG": DDPG, "TD3": TD3, "SAC": SAC, "A2C":A2C, "A2C_PINN":A2C_PINN}
    # SB3_ALGOS = {"PPO": PPO}   # add others when needed
    TRAIN_STEPS_MAP = {"PPO": 20_000, "PPO_PINN": 20_000, "TD3": 50_000, "TD3_PINN": 50_000,
                       "SAC": 50_000, "DDPG": 30_000, "DDPG_PINN": 30_000, "A2C": 20_000, "A2C_PINN": 20_000}

    results = {}
    metrics_rows = []

    # --- Train & Log SB3 models ------------------------------------------------
    for algo_name, Algo in SB3_ALGOS.items():
        print(f"\n=== Training {algo_name} on {name} ===")
        algo_root = os.path.join(port_root, algo_name)
        fig_dir = os.path.join(algo_root, "figs")
        log_dir = os.path.join(algo_root, "logs")
        _ensure_dir(fig_dir); _ensure_dir(log_dir)

        # TRAIN ENV (wrapped) + MODEL
        train_env = DummyVecEnv([lambda: Monitor(MultiAssetPortfolioEnv(train_df, assets, FEATURES, window_size))])
        if algo_name in ["PPO_PINN","TD3_PINN"]:
            model = Algo(env=train_env, seed=SEED, verbose=0)
        else:
            model = Algo(policy="MlpPolicy", env=train_env, seed=SEED, verbose=0)

        model.learn(total_timesteps=TRAIN_STEPS_MAP[algo_name])
        print(f"{algo_name} training done.")

        # --- Post-training rollouts to COLLECT logs on Train & Test ---
        # train logs
        train_env_eval = MultiAssetPortfolioEnv(train_df, assets, FEATURES, window_size)
        train_logs = rollout_collect(train_env_eval, model, deterministic=True)
        # test logs
        test_env_eval = MultiAssetPortfolioEnv(test_df, assets, FEATURES, window_size)
        test_logs = rollout_collect(test_env_eval, model, deterministic=True)

        # --- Save CSV logs (train/test rewards & weights) ---
        pd.DataFrame({"reward": train_logs["rewards"], "loss_proxy": train_logs["loss_proxy"]}) \
            .to_csv(os.path.join(log_dir, "train_rewards.csv"), index=False)
        pd.DataFrame(train_logs["weights"], columns=assets) \
            .to_csv(os.path.join(log_dir, "train_weights.csv"), index=False)
        pd.DataFrame({"reward": test_logs["rewards"]}) \
            .to_csv(os.path.join(log_dir, "test_rewards.csv"), index=False)

        # meta
        with open(os.path.join(log_dir, "meta.json"), "w") as f:
            json.dump({"assets": assets, "train_steps": int(TRAIN_STEPS_MAP[algo_name])}, f, indent=2)

        # --- Plots: reward & loss (train), reward (test), weights snapshots, wealth ---
        plot_reward_loss(train_logs["rewards"], train_logs["loss_proxy"],
                         os.path.join(fig_dir, "reward_loss_train.png"),
                         f"{algo_name} — Average reward & loss (train)")

        plot_reward_loss(test_logs["rewards"], np.zeros_like(test_logs["rewards"]),
                         os.path.join(fig_dir, "reward_loss_test.png"),
                         f"{algo_name} — Average reward (test)")

        plot_weight_snapshots(train_logs["weights"], assets,
                              os.path.join(fig_dir, "weights_snapshots.png"))
        plot_weight_snapshots(train_logs["weights"], assets,
                              os.path.join(fig_dir, "weights_snapshots2.png"), style="heatmap")
        wealth_test = test_logs["wealth"]
        results[algo_name] = wealth_test
        metrics_rows.append({"Algorithm": algo_name, **compute_metrics(wealth_test)})

        # also save wealth figure per-algo (solo)
        plot_wealth({algo_name: wealth_test},
                    os.path.join(fig_dir, "wealth_test.png"),
                    f"Portfolio Wealth ({name}) — {algo_name} (test)")

    # --- Benchmarks (pgportfolio) ---------------------------------------------
    test_close = np.vstack([test_df[f"close_{a}"].values for a in assets]).T
    pg_agents = {
        "UBAH": (UBAH, {}),
        "CRP": (CRP, {}),
        "OLMAR": (OLMAR, {"window": 5, "eps": 10}),
        "RMR": (RMR, {"eps": 10}),
        "PAMR": (PAMR, {"eps": 10, "C": 0.5})
    }
    for label, (cls, params) in pg_agents.items():
        print(f"Simulating {label} on {name}…")
        wealth = simulate_pg_agent(cls, test_close, **params)
        results[label] = wealth
        metrics_rows.append({"Algorithm": label, **compute_metrics(wealth)})

    # --- Combined wealth plot & metrics table ---------------------------------
    plot_wealth(results,
                os.path.join(port_root, "wealth_all_methods.png"),
                f"Portfolio Wealth ({name}) — Test Period")

    metrics_df = (
        pd.DataFrame(metrics_rows)
          .set_index("Algorithm")
          .reindex(["PPO_PINN","TD3_PINN","DDPG_PINN","PPO","DDPG","TD3","SAC",
                    "UBAH","CRP","OLMAR","RMR","PAMR","A2C","A2C_PINN"])
          .round(2)
    )
    csv_path = os.path.join(port_root, f"metrics_{name.replace(' ','_')}.csv")
    metrics_df.to_csv(csv_path)
    end_time = time.time()
    print(f"\n----------------  Performance Comparison: {name}  ----------------")
    print(metrics_df.to_string())
    print(f"Saved: {csv_path}")

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
            'assets': ["HPG.VN", "VIX.VN", "MWG.VN", "DIG.VN", "MSN.VN", "STB.VN", "VSC.VN", "HSG.VN", "FPT.VN", "EIB.VN"],
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
