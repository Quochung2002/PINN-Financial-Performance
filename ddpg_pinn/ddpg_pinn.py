from typing import Any, Optional, TypeVar, Union
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.common.noise import ActionNoise
from stable_baselines.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines.td3.policies import TD3Policy
from stable_baselines.td3.td3 import TD3
from stable_baselines.common.utils import polyak_update

SelfDDPG = TypeVar("SelfDDPG", bound="DDPG_PINN")


class SimpleLNN(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.V_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Potential energy
        )
        self.L_net = nn.Linear(state_dim, state_dim * state_dim)  # For mass matrix (lower triangular)

    def forward(self, q, q_dot, action):
        V = self.V_net(q)  # Potential
        L = self.L_net(q).view(-1, q.shape[1], q.shape[1])  # Lower triangular for M = L @ L.T
        M = L @ L.transpose(1, 2)
        predicted_accel = torch.linalg.solve(M, action.unsqueeze(-1)).squeeze(-1)  # Solve M * ddq = tau
        return predicted_accel, V  # For energy checks


class DDPG_PINN(TD3):
    """
    DDPG with a Physics-Informed term on the actor loss.
    Improvements: Enhanced adaptive weighting with variance consideration, optional Black-Scholes residual for finance,
    better normalization with separate EMAs for stability, and regime-detection proxy via volatility clustering.
    """

    def __init__(
        self,
        policy: Union[str, type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        physics_type: str = "newtons_laws",  # 'none', 'energy_conservation', 'newtons_laws', 'navier_stokes'
        lambda_phys: float = 0.1,
        mass: float = 1.0,
        gravity: float = 9.81,
        length: float = 1.0,
        viscosity: float = 0.01,  # Analogous to market friction
        density: float = 1.0,     # Asset "density"
        dt: float = 1.0,          # Use 1.0 for daily bars
        risk_free_rate: float = 0.05,  # For Black-Scholes
        volatility_penalty: float = 0.1,  # Penalty for high volatility regimes
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            # DDPG specifics: single critic, no delayed updates
            policy_delay=1,
            target_noise_clip=0.0,
            target_policy_noise=0.1,
            _init_setup_model=False,
        )

        self.physics_type = physics_type
        self.lambda_phys = lambda_phys
        self.mass = mass
        self.gravity = gravity
        self.length = length
        self.viscosity = viscosity
        self.density = density
        self.dt = dt
        self.risk_free_rate = risk_free_rate
        self.volatility_penalty = volatility_penalty

        # Enhanced normalization: Separate EMAs for mean and variance
        self._actor_mean_ema = 0.0
        self._actor_var_ema = 1.0
        self._phys_mean_ema = 0.0
        self._phys_var_ema = 1.0
        self._ema_beta = 0.995  # Smoother EMA for stability

        # Regime proxy: EMA volatility
        self._vol_ema = 0.0
        self._vol_beta = 0.9

        if _init_setup_model:
            self._setup_model()

    def _compute_physics_loss(self, observations: th.Tensor, actions: th.Tensor, next_observations: th.Tensor) -> th.Tensor:
        physics_loss = th.tensor(0.0, device=self.device)

        batch_size = observations.shape[0]
        n_assets = actions.shape[1]

        window_size = self.observation_space.shape[0]
        features_per_step = self.observation_space.shape[1]

        if n_assets == 0:
            return physics_loss

        n_features = features_per_step // n_assets

        try:
            obs_reshaped = observations.view(batch_size, window_size, n_assets, n_features)
            next_obs_reshaped = next_observations.view(batch_size, window_size, n_assets, n_features)
        except RuntimeError as e:
            if self.verbose > 0:
                print(f"Error reshaping physics tensors: {e}")
            return physics_loss

        # Compute returns (velocity analogy)
        returns = (next_obs_reshaped[:, -1, :, 3] - obs_reshaped[:, -1, :, 3]) / obs_reshaped[:, -1, :, 3]  # Assuming close at index 3

        # Update volatility EMA
        with th.no_grad():
            batch_vol = returns.std(dim=1).mean()
            self._vol_ema = self._vol_beta * self._vol_ema + (1 - self._vol_beta) * batch_vol.item()

        if self.physics_type == "black_scholes":
            # Black-Scholes PDE residual for option-like pricing in portfolio
            S = obs_reshaped[:, -1, :, 3]  # Prices
            dS_dt = returns / self.dt  # Approximate partial V/partial t ~ returns
            sigma = th.std(returns, dim=1, keepdim=True)  # Batch volatility

            # Use actor output as proxy for delta (hedge ratio)
            delta = actions  # Partial V/partial S ~ allocation

            # Second derivative approximation (finite diff across assets)
            d_delta_dS = (delta[:, 1:] - delta[:, :-1]) / (S[:, 1:] - S[:, :-1] + 1e-8)
            gamma = (d_delta_dS[:, 1:] + d_delta_dS[:, :-1]) / (S[:, :-2] + 1e-8)  # Approx d2V/dS2

            # BS residual
            bs_res = dS_dt + self.risk_free_rate * S[:, :-2] * delta[:, :-2] + 0.5 * sigma[:, :-2]**2 * S[:, :-2]**2 * gamma - self.risk_free_rate * actions[:, :-2]
            physics_loss = (bs_res ** 2).mean() + self.volatility_penalty * sigma.mean()**2  # Add vol penalty for regime

        elif self.physics_type == "navier_stokes":
            # Existing NS, with added Laplacian computation
            velocity = returns
            laplacian = th.zeros_like(velocity)
            for i in range(1, n_assets - 1):
                laplacian[:, i] = (velocity[:, i-1] - 2 * velocity[:, i] + velocity[:, i+1])

            ns_residual = velocity + self.dt * (actions - (self.viscosity / self.density) * laplacian)
            physics_loss = F.mse_loss(ns_residual, th.zeros_like(ns_residual)) + self.volatility_penalty * (velocity.var(dim=1).mean())

        elif self.physics_type == "energy_conservation":
            kinetic = 0.5 * self.mass * (returns ** 2).mean(dim=1)
            potential = self.gravity * self.length * (1 - th.cos(obs_reshaped[:, -1, :, 3]))  # Pendulum analogy
            delta_energy = kinetic.unsqueeze(1) + potential - (kinetic.unsqueeze(1) + potential).roll(1, dims=1)
            physics_loss = (delta_energy ** 2).mean()

        elif self.physics_type == "newtons_laws":
            acceleration = returns / self.dt
            predicted_accel = actions / self.mass
            physics_loss = F.mse_loss(predicted_accel, acceleration)

        # Clamp for stability
        physics_loss = th.clamp(physics_loss, 0.0, 10.0)

        return physics_loss

    def train(self, gradient_steps: int, batch_size: int) -> None:
        actor_losses = []
        critic_losses = []
        physics_losses = []

        self._n_updates += gradient_steps

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                next_actions = self.actor_target(replay_data.next_observations)
                next_q = self.critic_target(replay_data.next_observations, next_actions)[0]
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q

            current_q = self.critic(replay_data.observations, replay_data.actions)[0]
            critic_loss = F.mse_loss(current_q, target_q)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic.optimizer.step()

            # Actor update with improved PIRL
            actions_pi = self.actor(replay_data.observations)
            q1 = self.critic(replay_data.observations, actions_pi)[0]

            # Update actor EMA stats
            batch_mean = q1.mean().item()
            batch_var = q1.var().item()
            self._actor_mean_ema = self._ema_beta * self._actor_mean_ema + (1 - self._ema_beta) * batch_mean
            self._actor_var_ema = self._ema_beta * self._actor_var_ema + (1 - self._ema_beta) * batch_var

            # Normalized actor loss
            actor_loss = - (q1 - self._actor_mean_ema) / th.sqrt(th.tensor(self._actor_var_ema) + 1e-6).to(self.device)

            # Physics loss
            physics_loss_raw = self._compute_physics_loss(
                replay_data.observations, actions_pi, replay_data.next_observations
            )
            physics_losses.append(physics_loss_raw.item())

            # Update physics EMA stats
            self._phys_mean_ema = self._ema_beta * self._phys_mean_ema + (1 - self._ema_beta) * physics_loss_raw.item()
            self._phys_var_ema = self._ema_beta * self._phys_var_ema + (1 - self._ema_beta) * physics_loss_raw.var().item()

            # Normalized physics loss
            physics_loss = (physics_loss_raw - self._phys_mean_ema) / th.sqrt(th.tensor(self._phys_var_ema) + 1e-6).to(self.device)

            # Variance-aware adaptive lambda: Penalize high variance in physics for stability
            with th.no_grad():
                a_mag = actor_loss.abs().mean()
                p_mag = physics_loss.abs().mean()
                p_var_penalty = physics_loss.var().clamp_min(1e-6).sqrt()
                lambda_adapt = (a_mag / (p_mag + p_var_penalty)).clamp(0.1, 10.0) * self.lambda_phys

            total_actor_loss = actor_loss.mean() + lambda_adapt * physics_loss.mean()
            actor_losses.append(total_actor_loss.item())

            self.actor.optimizer.zero_grad()
            total_actor_loss.backward()
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor.optimizer.step()

            # Target updates
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        # Logging
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/physics_loss", np.mean(physics_losses))
            self.logger.record("train/lambda_adapt", lambda_adapt.item())
            self.logger.record("train/vol_ema", self._vol_ema)
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self: SelfDDPG,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DDPG_PINN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDDPG:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )