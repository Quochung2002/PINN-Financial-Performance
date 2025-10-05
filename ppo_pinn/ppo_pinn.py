import warnings
from typing import Any, ClassVar, Generator, NamedTuple, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor
from stable_baselines3.ppo.ppo import PPO

from stable_baselines.common.policies import ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from stable_baselines.common.type_aliases import RolloutReturn
from stable_baselines.ppo_pinn.policies_pinn import KANActorCriticPolicy

SelfPPOPINN = TypeVar("SelfPPOPINN", bound="PPO_PINN")

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
        next_obs: np.ndarray,
    ) -> None:
        self.next_observations[self.pos] = np.array(next_obs).copy()
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: Optional[int] = None) -> Generator[PhysicsRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Prepare vec indices
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        # Return everything, don't create minibatches
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

class PPO_PINN(PPO):
    """
    Proximal Policy Optimization (PPO) with Physics-Informed Neural Network (PINN) enhancements.

    Extends PPO by using a KAN-based policy and incorporating a physics-informed loss in the update.

    :param policy: The policy model to use ('KANMlpPolicy', MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from
    :param learning_rate: The learning rate
    :param n_steps: Number of steps per update
    :param batch_size: Minibatch size
    :param n_epochs: Number of epochs for optimization
    :param gamma: Discount factor
    :param gae_lambda: GAE lambda
    :param clip_range: Clipping parameter
    :param clip_range_vf: Value function clipping
    :param normalize_advantage: Normalize advantages
    :param ent_coef: Entropy coefficient
    :param vf_coef: Value function coefficient
    :param max_grad_norm: Max gradient norm
    :param use_sde: Use gSDE
    :param sde_sample_freq: SDE sample frequency
    :param rollout_buffer_class: Custom rollout buffer (uses PhysicsRolloutBuffer by default)
    :param rollout_buffer_kwargs: Kwargs for rollout buffer
    :param target_kl: Target KL divergence
    :param stats_window_size: Stats window size
    :param tensorboard_log: Tensorboard log dir
    :param policy_kwargs: Policy kwargs
    :param verbose: Verbosity
    :param seed: Seed
    :param device: Device
    :param _init_setup_model: Init model
    :param physics_type: Physics constraint type ('none', 'energy_conservation', 'newtons_laws', 'navier_stokes')
    :param lambda_phys: Physics loss weight
    :param viscosity: Viscosity for Navier-Stokes
    :param density: Density for Navier-Stokes
    :param dt: Time step
    """
    policy_aliases: ClassVar[dict[str, type[ActorCriticPolicy]]] = {
    "MlpPolicy": ActorCriticPolicy,
    "CnnPolicy": ActorCriticCnnPolicy,
    "MultiInputPolicy": MultiInputActorCriticPolicy,
    "KANMlpPolicy": KANActorCriticPolicy,  # Add this line
    }
    
    def __init__(
        self,
        env: Union[GymEnv, str],
        policy: Union[str, type[ActorCriticPolicy]] = "KANMlpPolicy",
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[PhysicsRolloutBuffer]] = PhysicsRolloutBuffer,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        physics_type: str = "newtons_laws",#"navier_stokes",
        lambda_phys: float = 0.1,
        viscosity: float = 0.01,
        density: float = 1.0,
        dt: float = 0.05,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            n_steps,
            batch_size,
            n_epochs,
            gamma,
            gae_lambda,
            clip_range,
            clip_range_vf,
            normalize_advantage,
            ent_coef,
            vf_coef,
            max_grad_norm,
            use_sde,
            sde_sample_freq,
            rollout_buffer_class,
            rollout_buffer_kwargs,
            target_kl,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )
        self.physics_type = physics_type
        self.lambda_phys = lambda_phys
        self.viscosity = viscosity
        self.density = density
        self.dt = dt

    def collect_rollouts(
        self,
        env: GymEnv,
        callback: MaybeCallback,
        n_episodes: int = 1,
        n_rollout_steps: int = -1,
    ) -> RolloutReturn:
        self.policy.set_training_mode(False)

        n_steps = self.n_steps if n_rollout_steps == -1 else n_rollout_steps

        if self.rollout_buffer.full:
            self.rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        rollout_steps = 0
        n_collected_episodes = 0
        continue_training = True

        while rollout_steps < n_steps:
            if self.use_sde and self.sde_sample_freq > 0 and rollout_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            # --- FIX #1: Sum log probabilities during data collection ---
            if isinstance(self.action_space, spaces.Box):
                if len(log_probs.shape) > 1:
                    log_probs = log_probs.sum(axis=1)
            # --- END FIX ---
            actions = actions.cpu().numpy()

            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return RolloutReturn(rollout_steps * env.num_envs, n_collected_episodes, continue_training=False)

            self._update_info_buffer(infos, dones)

            self.rollout_buffer.add(
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
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            n_collected_episodes += sum(dones)

            rollout_steps += 1

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        self.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return RolloutReturn(rollout_steps * env.num_envs, n_collected_episodes, continue_training)

    def _compute_physics_loss(self, observations: th.Tensor, actions: th.Tensor,
                              next_observations: th.Tensor) -> th.Tensor:
        physics_loss = th.tensor(0.0, device=self.device)

        if self.physics_type == "none":
            return physics_loss

        # --- FIX: Correctly reshape the flattened observation tensor ---
        batch_size = observations.shape[0]
        n_assets = actions.shape[1]

        # Get original dimensions from the environment's observation space
        window_size = self.observation_space.shape[0]
        features_per_step = self.observation_space.shape[1]

        # Ensure n_assets is not zero to avoid division error
        if n_assets == 0:
            return physics_loss  # Or handle as an error

        n_features = features_per_step // n_assets

        # Reshape the flattened tensors back to their structured view
        # (batch_size, window_size, n_assets, n_features)
        try:
            obs_reshaped = observations.view(batch_size, window_size, n_assets, n_features)
            next_obs_reshaped = next_observations.view(batch_size, window_size, n_assets, n_features)
        except RuntimeError as e:
            if self.verbose > 0:
                print(f"Error reshaping physics tensors: {e}")
            return physics_loss
        # --- END FIX ---

        # Compute velocity per feature
        velocity = (next_obs_reshaped - obs_reshaped) / self.dt

        if self.physics_type == "energy_conservation":
            # Mean over features for per-asset
            obs_mean = obs_reshaped.mean(dim=3)
            predicted_next_mean = obs_mean + self.dt * actions.unsqueeze(1)
            next_mean = next_obs_reshaped.mean(dim=3)
            physics_loss = F.mse_loss(predicted_next_mean, next_mean)


        # In ppo_pinn.py, inside the _compute_physics_loss method...

        elif self.physics_type == "newtons_laws":
            # The 'velocity' tensor has shape [batch, window_size, n_assets, n_features].
            # We must only use the velocity from the LAST time step to match the action.
            # 1. Isolate the velocity from the most recent time step.
            # Shape changes from [64, 5, 10, 17] to [64, 10, 17].
            last_step_velocity = velocity[:, -1, :, :]
            # 2. Average across features to get a single acceleration value per asset.
            # Shape becomes [64, 10].
            accel = last_step_velocity.mean(dim=2)
            # 3. Now, both 'actions' and 'accel' have the same shape [64, 10].
            # We can compare them directly without unsqueezing.

            physics_loss = F.mse_loss(actions, accel)


        elif self.physics_type == "navier_stokes":
            # Placeholder laplacian (zeros); mean over features
            laplacian = th.zeros_like(velocity)
            velocity_mean = velocity.mean(dim=3)
            laplacian_mean = laplacian.mean(dim=3)
            actions_unsq = actions.unsqueeze(1)
            ns_residual = velocity_mean + self.dt * (actions_unsq - (self.viscosity / self.density) * laplacian_mean)
            physics_loss = F.mse_loss(ns_residual, th.zeros_like(ns_residual))

        return physics_loss

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # --- FIX: Update schedules and handle log_prob shapes ---

        # Switch to train mode (this affects dropout and batch norm)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        physics_losses = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                with th.no_grad():
                    features = self.policy.extract_features(rollout_data.observations)
                    latent_pi = self.policy.mlp_extractor.forward_actor(features)
                    mean_actions = self.policy.action_net(latent_pi)

                physics_loss = self._compute_physics_loss(
                    rollout_data.observations, mean_actions, rollout_data.next_observations
                )
                physics_losses.append(physics_loss.item())

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

                # Sum log probabilities for continuous action spaces
                if isinstance(self.action_space, spaces.Box):
                    if len(log_prob.shape) > 1:
                        log_prob = log_prob.sum(axis=1)

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped policy loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae) targets
                value_loss = F.mse_loss(values_pred.flatten(), rollout_data.returns)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.lambda_phys * physics_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/physics_loss", np.mean(physics_losses))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)