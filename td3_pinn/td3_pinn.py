from typing import Any, ClassVar, Optional, TypeVar, Union
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.td3.td3 import TD3
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines.td3_pinn.policies_pinn import KANTD3Policy

SelfTD3 = TypeVar("SelfTD3", bound="TD3_PINN")


class TD3_PINN(TD3):
    """
    Twin Delayed DDPG (TD3) with Physics-Informed Neural Network (PINN) enhancements.
    
    Original TD3 Paper: https://arxiv.org/abs/1802.09477
    DDPG Paper: https://arxiv.org/abs/1509.02971
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
    
    Extends TD3 by incorporating a physics-informed loss in the actor update,
    such as energy conservation for pendulum-like environments, Newton's laws for robotics,
    or Navier-Stokes for financial market analogies.
    
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, KANMlpPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: Learning rate for adam optimizer
    :param buffer_size: Size of the replay buffer
    :param learning_starts: Steps to collect transitions before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: Soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: Discount factor
    :param train_freq: Update frequency, can be int or tuple (e.g., (5, "step"))
    :param gradient_steps: Number of gradient steps per rollout
    :param action_noise: Action noise for exploration
    :param replay_buffer_class: Custom replay buffer class
    :param replay_buffer_kwargs: Keyword args for replay buffer
    :param optimize_memory_usage: Enable memory-efficient replay buffer
    :param policy_delay: Policy update frequency relative to critic updates
    :param target_policy_noise: Standard deviation of target policy smoothing noise
    :param target_noise_clip: Limit for target policy smoothing noise
    :param physics_type: Type of physics constraint ('none', 'energy_conservation', 'newtons_laws', 'navier_stokes', 'black_scholes')
    :param lambda_phys: Weight for physics-informed loss
    :param mass: Mass parameter for physics calculations
    :param gravity: Gravity constant for pendulum energy
    :param length: Pendulum length for energy calculation
    :param viscosity: Viscosity for Navier-Stokes analogy
    :param density: Density for Navier-Stokes analogy
    :param dt: Time step for acceleration calculations
    :param risk_free_rate: Risk-free rate for Black-Scholes
    :param volatility_penalty: Penalty for high volatility regimes
    :param tensorboard_log: Log location for tensorboard
    :param policy_kwargs: Additional arguments for policy creation
    :param verbose: Verbosity level (0: none, 1: info, 2: debug)
    :param seed: Seed for pseudo-random generators
    :param device: Device to run on (cpu, cuda, auto)
    :param _init_setup_model: Whether to build the network on instantiation
    """
    
    policy_aliases: ClassVar[dict[str, type[TD3Policy]]] = {
        **TD3.policy_aliases,
        "KANMlpPolicy": KANTD3Policy,
        "KANCnnPolicy": KANTD3Policy,
        "KANMultiInputPolicy": KANTD3Policy,
    }

    def __init__(
        self,
        env: Union[GymEnv, str],
        policy: Union[str, type[TD3Policy]] = "KANMlpPolicy",
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
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        physics_type: str = "newtons_laws",
        lambda_phys: float = 0.1,
        mass: float = 1.0,
        gravity: float = 9.81,
        length: float = 1.0,
        viscosity: float = 0.01,
        density: float = 1.0,
        dt: float = 1.0,
        risk_free_rate: float = 0.05,
        volatility_penalty: float = 0.1,
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
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            optimize_memory_usage=optimize_memory_usage,
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

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

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
        returns = (next_obs_reshaped[:, -1, :, 3] - obs_reshaped[:, -1, :, 3]) / (obs_reshaped[:, -1, :, 3] + 1e-8)  # Assuming close at index 3

        # Update volatility EMA
        with th.no_grad():
            batch_vol = returns.std(dim=1).mean()
            self._vol_ema = self._vol_beta * self._vol_ema + (1 - self._vol_beta) * batch_vol.item()

        if self.physics_type == "black_scholes":
            # Black-Scholes PDE residual for option-like pricing in portfolio
            S = obs_reshaped[:, -1, :, 3]  # Prices
            dS_dt = returns / self.dt  # Approximate partial V/partial t ~ returns
            sigma = th.std(returns, dim=1)  # Batch volatility, shape (b,)

            # Use actor output as proxy for delta (hedge ratio)
            delta = actions  # Partial V/partial S ~ allocation

            # Second derivative approximation (finite diff across assets)
            d_delta_dS = (delta[:, 1:] - delta[:, :-1]) / (S[:, 1:] - S[:, :-1] + 1e-8)
            gamma = (d_delta_dS[:, 1:] - d_delta_dS[:, :-1]) / ((S[:, 2:] - S[:, :-2]) / 2 + 1e-8)  # Approx d2V/dS2, fixed to -

            # BS residual
            bs_res = dS_dt[:, :-2] + self.risk_free_rate * S[:, :-2] * delta[:, :-2] + 0.5 * sigma.unsqueeze(1).expand(-1, n_assets-2)**2 * S[:, :-2]**2 * gamma - self.risk_free_rate * actions[:, :-2]
            physics_loss = (bs_res ** 2).mean(dim=1) + self.volatility_penalty * sigma**2  # per batch

        elif self.physics_type == "navier_stokes":
            # Existing NS, with added Laplacian computation
            velocity = returns
            laplacian = th.zeros_like(velocity)
            for i in range(1, n_assets - 1):
                laplacian[:, i] = (velocity[:, i-1] - 2 * velocity[:, i] + velocity[:, i+1])

            ns_residual = velocity + self.dt * (actions - (self.viscosity / self.density) * laplacian)
            physics_loss = F.mse_loss(ns_residual, th.zeros_like(ns_residual), reduction='none').mean(dim=1) + self.volatility_penalty * (velocity.var(dim=1))

        elif self.physics_type == "energy_conservation":
            kinetic = 0.5 * self.mass * (returns ** 2).mean(dim=1)
            potential = self.gravity * self.length * (1 - th.cos(obs_reshaped[:, -1, :, 3]))  # Pendulum analogy
            delta_energy = kinetic.unsqueeze(1) + potential - (kinetic.unsqueeze(1) + potential).roll(1, dims=1)
            physics_loss = (delta_energy ** 2).mean(dim=1)

        elif self.physics_type == "newtons_laws":
            acceleration = returns / self.dt
            predicted_accel = actions / self.mass
            physics_loss = F.mse_loss(predicted_accel, acceleration, reduction='none').mean(dim=1)

        # Clamp for stability
        physics_loss = th.clamp(physics_loss, 0.0, 10.0)

        return physics_loss

    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        # Adapted from parent TD3 train method with enhanced physics integration
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses = []
        critic_losses = []
        physics_losses = []

        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Target policy smoothing
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute target Q-values
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q = th.min(next_q_values, dim=1, keepdim=True)[0]
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q

            # Critic update
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(F.mse_loss(current_q, target_q) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic.optimizer.step()

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)

            if self._n_updates % self.policy_delay == 0:
                # Actor update with improved PIRL
                actions_pi = self.actor(replay_data.observations)
                q1 = self.critic.q1_forward(replay_data.observations, actions_pi)

                # Update actor EMA stats
                batch_mean = q1.mean().item()
                batch_var = q1.var().item()
                self._actor_mean_ema = self._ema_beta * self._actor_mean_ema + (1 - self._ema_beta) * batch_mean
                self._actor_var_ema = self._ema_beta * self._actor_var_ema + (1 - self._ema_beta) * batch_var

                # Normalized actor loss
                actor_loss = - (q1 - self._actor_mean_ema) / (th.sqrt(th.tensor(self._actor_var_ema, device=self.device)) + 1e-6)

                # Physics loss
                physics_loss_raw = self._compute_physics_loss(
                    replay_data.observations, actions_pi, replay_data.next_observations
                )
                physics_losses.append(physics_loss_raw.mean().item())

                # Update physics EMA stats
                phys_mean = physics_loss_raw.mean().item()
                phys_var = physics_loss_raw.var().item() if batch_size > 1 else 0.0
                self._phys_mean_ema = self._ema_beta * self._phys_mean_ema + (1 - self._ema_beta) * phys_mean
                self._phys_var_ema = self._ema_beta * self._phys_var_ema + (1 - self._ema_beta) * phys_var

                # Normalized physics loss
                physics_loss_norm = (physics_loss_raw - self._phys_mean_ema) / (th.sqrt(th.tensor(self._phys_var_ema, device=self.device)) + 1e-6)

                # Variance-aware adaptive lambda: Penalize high variance in physics for stability
                with th.no_grad():
                    a_mag = actor_loss.abs().mean()
                    p_mag = physics_loss_norm.abs().mean()
                    p_var_penalty = physics_loss_norm.var().clamp_min(1e-6).sqrt()
                    lambda_adapt = (a_mag / (p_mag + p_var_penalty)).clamp(0.1, 10.0) * self.lambda_phys

                total_actor_loss = actor_loss.mean() + lambda_adapt * physics_loss_norm.mean()
                actor_losses.append(total_actor_loss.item())

                self.actor.optimizer.zero_grad()
                total_actor_loss.backward()
                th.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                self.actor.optimizer.step()

                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        # Logging
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/physics_loss", np.mean(physics_losses))
            self.logger.record("train/lambda_adapt", lambda_adapt.item())
            self.logger.record("train/vol_ema", self._vol_ema)
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self: SelfTD3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TD3_PINN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTD3:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )