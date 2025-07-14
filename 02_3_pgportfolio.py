# ============================================================================
#  Multi‑Asset Portfolio Management ― PPO • DDPG • TD3 • SAC + pgportfolio BL
# ============================================================================

import os, logging
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from io import StringIO

# ---------- monkey‑patch for NumPy 2.x (pgportfolio uses np.alltrue) ----------
if not hasattr(np, "alltrue"):
    np.alltrue = np.all

# ---------- SB3 imports ------------------------------------------------------
from stable_baselines import PPO, TD3, SAC
from stable_baselines.ddpg_pinn.ddpg_pinn import DDPG_PINN as DDPG
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.monitor import Monitor

# # ---------- suppress TensorFlow clutter (pgportfolio) ------------------------
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# logging.getLogger("tensorflow").setLevel(logging.FATAL)

# ---------- pgportfolio online agents ---------------------------------------
from pgportfolio.tdagent.algorithms.olmar import OLMAR
from pgportfolio.tdagent.algorithms.rmr import RMR
from pgportfolio.tdagent.algorithms.pamr import PAMR
from pgportfolio.tdagent.algorithms.crp import CRP
from pgportfolio.tdagent.algorithms.ubah import UBAH
import random, torch
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
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

    # ---------------------------------------------------------------------
    def _get_asset_window(self, asset: str, col: str):
        series = self.df[f"{col}_{asset}"].values[
            self.pointer - self.window_size : self.pointer
        ]
        return np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_observation(self):
        obs = [
            self._get_asset_window(asset, col)
            for asset in self.assets
            for col in ("open", "high", "low", "close")
        ]
        return np.nan_to_num(np.array(obs).T, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pointer = self.window_size
        self.weights = np.ones(self.n_assets) / self.n_assets
        return self._get_observation(), {}

    # ---------------------------------------------------------------------
    def step(self, action):
        # --- enforce simplex ---------------------------------------------
        action = np.clip(action, 0.0, 1.0)
        action = action / action.sum() if action.sum() > 0 else np.ones_like(action) / self.n_assets
        self.weights = action.copy()

        # --- price relatives --------------------------------------------
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
#                        SB3 SIMULATION HELPER
# ----------------------------------------------------------------------------
def simulate_sb3_strategy(env: MultiAssetPortfolioEnv, model):
    obs, _ = env.reset()
    wealth = [1000.0]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        wealth.append(wealth[-1] * (1 + reward))
    return wealth


# ----------------------------------------------------------------------------
#                     pgportfolio AGENT SIMULATION
# ----------------------------------------------------------------------------
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

        new_b = agent.decide_by_history(rel, last_b)
        if new_b.shape != (n_assets,):
            print(f"[WARN] {agent_cls.__name__}: weight shape {new_b.shape}, correcting.")
            new_b = np.ones(n_assets) / n_assets

        wealth.append(wealth[-1] * (1 + np.dot(new_b, rel - 1)))
        last_b = new_b
    return wealth


# ----------------------------------------------------------------------------
#                            METRICS
# ----------------------------------------------------------------------------
def compute_metrics(wealth, *, freq=252, risk_free_rate=0.0):
    """
    Parameters
    ----------
    wealth : 1‑D array‑like
        Path of portfolio value (≥ 2 points).
    freq : int, default 252
        Number of trading periods per year.
    risk_free_rate : float, default 0.0
        Annual risk‑free rate as a decimal (e.g. 0.02 for 2 %).
    Returns
    -------
    dict with:
      'APV'              final wealth
      'Ann.Return(%)'    geometric annualised return
      'Ann.Vol(%)'       annualised volatility
      'Sharpe'           (µ – rf) / σ
      'MDD(%)'           maximum drawdown
      'Calmar'           ann.return / |MDD|
    """
    wealth = np.asarray(wealth, dtype=float)
    if wealth.size < 2:
        raise ValueError("Need at least 2 wealth points")

    # --------------------------------------------------------------------
    # daily log returns for numerical stability (optional but recommended)
    rets = np.diff(wealth) / wealth[:-1]

    # geometric annualised return
    ann_return = (wealth[-1] / wealth[0]) ** (freq / rets.size) - 1.0

    # annualised volatility (sample std * √freq)
    vol_daily = np.std(rets, ddof=1)
    ann_vol = vol_daily * np.sqrt(freq)

    # Sharpe ratio
    excess_ann_return = ann_return - risk_free_rate
    sharpe = excess_ann_return / ann_vol if ann_vol > 0 else np.nan

    # maximum drawdown
    running_max = np.maximum.accumulate(wealth)
    drawdown = (wealth - running_max) / running_max
    mdd = drawdown.min()  # negative number

    # Calmar ratio
    calmar = ann_return / abs(mdd) if mdd != 0 else np.nan

    return {
        "APV": wealth[-1],
        "Ann.Return(%)": ann_return * 100,
        "Ann.Vol(%)": ann_vol * 100,
        "Sharpe": sharpe,
        "MDD(%)": mdd * 100,
        "Calmar": calmar,
    }


# ============================================================================
#                                  MAIN
# ============================================================================
if __name__ == "__main__":

    # -----------------------  load & prep data  ---------------------------
    df = pd.read_csv("portfolio_data.csv", parse_dates=["date"], dayfirst=True)
    df = df.set_index("date").ffill().bfill()

    assets = sorted(
        {"_".join(col.split("_")[1:]) for col in df.columns if col.startswith("close_")}
    )
    print(f"Assets ({len(assets)}):", assets)

    # ----------------------  train / test split  --------------------------
    train_df = df["2010-01-01":"2022-12-31"].reset_index(drop=True)
    test_df = df["2023-01-01":"2024-12-31"].reset_index(drop=True)

    window_size = 5

    # ----------------------  RL algorithms dict  -------------------------
    SB3_ALGOS = {
        "PPO": PPO,
        "DDPG": DDPG,
        "TD3": TD3,
        "SAC": SAC,
    }

    results = {}
    metrics_rows = []

    for name, Algo in SB3_ALGOS.items():
        print(f"\n=== Training {name} ===")
        train_env = DummyVecEnv(
            [lambda: Monitor(MultiAssetPortfolioEnv(train_df, assets, window_size))]
        )
        # sensible defaults; tweak if desired
        if name == "PPO":  # keep *exactly* the old rollout size
            model = Algo(
                "MlpPolicy", train_env,
                seed=SEED, verbose=1,
                n_steps=1024,  # << same as earlier script
                batch_size=64,
                learning_rate=3e-4,
            )
        else:  # DDPG / TD3 / SAC keep defaults
            model = Algo("MlpPolicy", train_env, seed=SEED, verbose=1)
        model.learn(total_timesteps=20_000)
        print(f"{name} training done.")

        # evaluation
        eval_env = MultiAssetPortfolioEnv(test_df, assets, window_size)
        wealth = simulate_sb3_strategy(eval_env, model)
        results[name] = wealth
        row = {"Algorithm": name, **compute_metrics(wealth)}
        metrics_rows.append(row)

    # --------------------  pgportfolio baselines  ------------------------
    test_close = np.vstack([test_df[f"close_{a}"].values for a in assets]).T

    pg_agents = {
        "UBAH": (UBAH, {}),
        "CRP": (CRP, {}),
        "OLMAR": (OLMAR, {"window": 5, "eps": 10}),
        "RMR": (RMR, {"eps": 10}),
        "PAMR": (PAMR, {"eps": 10, "C": 0.5}),
    }

    for label, (cls, params) in pg_agents.items():
        print(f"Simulating {label} …")
        wealth = simulate_pg_agent(cls, test_close, **params)
        results[label] = wealth
        metrics_rows.append({"Algorithm": label, **compute_metrics(wealth)})

    # --------------------  BEST oracle  -----------------------------------
    print("Simulating BEST (oracle) …")
    base = test_close[0]
    final = test_close[-1]
    base[base == 0] = 1e-10
    best_idx = np.argmax(final / base)
    oracle_series = test_close[:, best_idx]
    oracle_series[oracle_series == 0] = 1e-10
    results["BEST"] = 1000.0 * oracle_series / oracle_series[0]
    metrics_rows.append(
        {"Algorithm": "BEST", **compute_metrics(results["BEST"])}
    )

    # --------------------  metrics table  ---------------------------------
    metrics_df = (
        pd.DataFrame(metrics_rows)
        .set_index("Algorithm")
        .loc[
            [
                "PPO",
                "DDPG",
                # "TD3",
                # "SAC",
                "UBAH",
                "CRP",
                "OLMAR",
                "RMR",
                "PAMR",
                "BEST",
            ]
        ]
        .round({"APV": 2, "SR(%)": 2, "CR": 2})
    )

    print("\n----------------  Performance Comparison  ----------------")
    print(
        metrics_df.to_string(
            formatters={
                "APV": "${:,.2f}".format,
                "SR(%)": "{:,.2f}%".format,
                "CR": "{:,.2f}%".format,
            }
        )
    )

    # --------------------  plot  -----------------------------------------
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(16, 8))
    for name, wealth in results.items():
        ls = "--" if name in {"UBAH", "CRP", "BEST"} else "-"
        # lw = 2.5 if name in {"PPO", "DDPG", "TD3", "SAC"} else 1.8
        lw = 2.5 if name in {"PPO", "DDPG"} else 1.8
        plt.plot(wealth, label=name, linestyle=ls, linewidth=lw)
    plt.title("Portfolio Wealth — Test 2021‑2023", fontsize=16)
    plt.xlabel("Trading Days", fontsize=12)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.show()
