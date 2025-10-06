from typing import Any, ClassVar, Dict, Generator, List, NamedTuple, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.a2c.a2c import A2C

from stable_baselines3.common.policies import ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from stable_baselines.a2c_pinn.policies_pinn import KANActorCriticPolicy

SelfA2CPINN = TypeVar("SelfA2CPINN", bound="A2C_PINN")

class PhysicsRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    next_observations: th.Tensor

class PhysicsRolloutBuffer(RolloutBuffer):
    """
    Custom rollout buffer that stores next_observations for physics-informed loss.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        next_obs: np.ndarray = None,  # Optional, but required for physics
    ) -> None:
        if next_obs is not None:
            self.next_observations[self.pos] = np.array(next_obs).copy()
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: Optional[int] = None) -> Generator[PhysicsRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> PhysicsRolloutBufferSamples:
        data = super()._get_samples(batch_inds)
        next_obs = self.to_torch(self.next_observations.reshape((-1, *self.obs_shape))[batch_inds])
        return PhysicsRolloutBufferSamples(*data, next_observations=next_obs)

class A2C_PINN(A2C):
    """
    Advantage Actor Critic (A2C) with Physics-Informed Neural Network (PINN) enhancements.
    Improvements: Dynamic scaling for physics loss, variance-weighted loss, Navier-Stokes and Newton's Laws for finance.
    """

    policy_aliases: ClassVar[dict[str, type[ActorCriticPolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
        "KANMlpPolicy": KANActorCriticPolicy,
    }

    def __init__(
        self,
        env: Union[GymEnv, str],
        policy: Union[str, type[ActorCriticPolicy]] = "KANMlpPolicy",
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 2048,  # Increased for better stability
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[PhysicsRolloutBuffer]] = PhysicsRolloutBuffer,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        normalize_advantage: bool = True,  # Enabled for better normalization
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        physics_type: str = "newtons_laws",  # Default to 'newtons_laws'; can switch to 'navier_stokes'
        lambda_phys: float = 0.01,  # Lowered initial value
        viscosity: float = 0.001,  # Lower for volatile markets
        density: float = 1.0,
        dt: float = 1.0,  # Adjusted for daily trading
        mass: float = 1.0,  # For Newton's Laws
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            rms_prop_eps=rms_prop_eps,
            use_rms_prop=use_rms_prop,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            normalize_advantage=normalize_advantage,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self.physics_type = physics_type
        self.lambda_phys = lambda_phys
        self.viscosity = viscosity
        self.density = density
        self.dt = dt
        self.mass = mass

    def collect_rollouts(
        self,
        env: GymEnv,
        callback: MaybeCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        for _ in range(n_rollout_steps):
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)

            actions = actions.cpu().numpy()

            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                callback.on_rollout_end()
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                next_obs=new_obs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def _compute_physics_loss(self, observations: th.Tensor, actions: th.Tensor, next_observations: th.Tensor) -> th.Tensor:
        physics_loss = th.tensor(0.0, device=self.device)

        batch_size = observations.shape[0]
        n_assets = actions.shape[1]

        window_size = self.observation_space.shape[0]
        features_per_step = self.observation_space.shape[1]

        if n_assets < 3:  # Need at least 3 assets for finite differences
            return physics_loss

        n_features = features_per_step // n_assets

        try:
            obs_reshaped = observations.view(batch_size, window_size, n_assets, n_features)
            next_obs_reshaped = next_observations.view(batch_size, window_size, n_assets, n_features)
        except RuntimeError as e:
            if self.verbose > 0:
                print(f"Error reshaping physics tensors: {e}")
            return physics_loss

        # Normalize observations for stability
        obs_mean = obs_reshaped.mean()
        obs_std = obs_reshaped.std() + 1e-8
        obs_reshaped = (obs_reshaped - obs_mean) / obs_std
        next_obs_reshaped = (next_obs_reshaped - obs_mean) / obs_std

        velocity = (next_obs_reshaped - obs_reshaped) / self.dt
        velocity = (velocity - velocity.mean()) / (velocity.std() + 1e-8)  # Z-score velocity

        if self.physics_type == "navier_stokes":
            close_t = obs_reshaped[:, -1, :, 3]
            close_t_minus_1 = obs_reshaped[:, -2, :, 3]
            u = (close_t - close_t_minus_1) / self.dt

            next_close_t = next_obs_reshaped[:, -1, :, 3]
            next_close_t_minus_1 = next_obs_reshaped[:, -2, :, 3]
            u_next = (next_close_t - next_close_t_minus_1) / self.dt

            du_dx = (u[:, 1:] - u[:, :-1]) / 1.0
            du2_dx2 = (du_dx[:, 1:] - du_dx[:, :-1]) / 1.0
            dp_dx = (actions[:, 1:] - actions[:, :-1]) / self.density / 1.0

            continuity_res = du_dx.abs().mean(dim=1)

            du_dt = (u_next[:, :-2] - u[:, :-2]) / self.dt
            conv_term = u[:, :-2] * du_dx[:, :-1]
            visc_term = self.viscosity * du2_dx2
            press_term = dp_dx[:, :-1]
            force_term = actions[:, :-2]

            momentum_res = du_dt + conv_term + press_term - visc_term - force_term
            # Fixed: Mean and var consistently
            physics_loss = (continuity_res ** 2).mean() + (momentum_res ** 2).mean() + (continuity_res ** 2).var() + (momentum_res ** 2).var(dim=1).mean()

        elif self.physics_type == "newtons_laws":
            # Newton's Laws: F = m * a, with actions as forces, velocity changes as acceleration
            acceleration = velocity[:, -1, :, :] / self.dt  # [batch, assets, features]
            predicted_accel = actions.unsqueeze(2) / self.mass  # [batch, assets, 1] to broadcast
            physics_loss = F.mse_loss(predicted_accel, acceleration.mean(dim=2).unsqueeze(2))  # Mean over features, unsqueeze for dim match
            physics_loss += physics_loss.var()  # Add variance for weighting

        else:
            actions_pi = self.policy(obs_tensor=observations)[0]
            actions_pi_next = self.policy(obs_tensor=next_observations)[0]
            physics_loss = F.mse_loss(actions_pi, actions_pi_next)

        # Clip for stability
        physics_loss = th.clamp(physics_loss, -10, 10)

        return physics_loss

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        entropy_losses = []
        pg_losses = []
        value_losses = []
        physics_losses = []

        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            with th.no_grad():
                features = self.policy.extract_features(rollout_data.observations)
                latent_pi = self.policy.mlp_extractor.forward_actor(features)
                mean_actions = self.policy.action_net(latent_pi)

            physics_loss = self._compute_physics_loss(
                rollout_data.observations, mean_actions, rollout_data.next_observations
            )
            physics_losses.append(physics_loss.item())

            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()
            
            if isinstance(self.action_space, spaces.Box):
                if len(log_prob.shape) > 1:
                    log_prob = log_prob.sum(dim=1)
                if entropy is not None and len(entropy.shape) > 1:
                    entropy = entropy.sum(dim=1)

            advantages = rollout_data.advantages
            if self.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(advantages * log_prob).mean()

            value_loss = F.mse_loss(rollout_data.returns, values)

            if entropy is None:
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            # Dynamic scaling
            with th.no_grad():
                policy_loss_magnitude = th.abs(policy_loss)
                physics_loss_magnitude = th.abs(physics_loss)
                scale_factor = policy_loss_magnitude / (physics_loss_magnitude + 1e-8)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.lambda_phys * scale_factor * physics_loss

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            entropy_losses.append(entropy_loss.item())
            pg_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/physics_loss", np.mean(physics_losses))
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def learn(
        self: SelfA2CPINN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "A2C_PINN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfA2CPINN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )