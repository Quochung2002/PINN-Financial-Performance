from typing import Any, Optional, TypeVar, Union
import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.common.noise import ActionNoise
from stable_baselines.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines.td3.policies import TD3Policy
from stable_baselines.td3.td3 import TD3
from stable_baselines.common.utils import polyak_update

SelfDDPG = TypeVar("SelfDDPG", bound="DDPG_PINN")


class DDPG_PINN(TD3):
    """
    Deep Deterministic Policy Gradient (DDPG) with Physics-Informed Neural Network (PINN) enhancements.

    Deterministic Policy Gradient: http://proceedings.mlr.press/v32/silver14.pdf
    DDPG Paper: https://arxiv.org/abs/1509.02971
    Introduction to DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html

    This implementation extends DDPG by incorporating a physics-informed loss in the actor update,
    such as energy conservation for pendulum-like environments or Newton's laws for robotics.
    The specific physics loss can be toggled via the `physics_type` parameter.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: Learning rate for adam optimizer, can be a function of progress (1 to 0)
    :param buffer_size: Size of the replay buffer
    :param learning_starts: Steps to collect transitions before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: Soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: Discount factor
    :param train_freq: Update frequency, can be int or tuple (e.g., (5, "step"))
    :param gradient_steps: Number of gradient steps per rollout, -1 for as many as env steps
    :param action_noise: Action noise type for exploration (e.g., from common.noise)
    :param replay_buffer_class: Custom replay buffer class (e.g., HerReplayBuffer)
    :param replay_buffer_kwargs: Keyword args for replay buffer creation
    :param optimize_memory_usage: Enable memory-efficient replay buffer variant
    :param physics_type: Type of physics constraint ('none', 'energy_conservation', 'newtons_laws')
    :param lambda_phys: Weight for the physics-informed loss (default: 0.1)
    :param mass: Mass parameter for physics calculations (e.g., for pendulum or robotics)
    :param gravity: Gravity constant for pendulum energy (default: 9.81)
    :param length: Pendulum length for energy calculation (default: 1.0)
    :param dt: Time step for acceleration calculations in Newton's laws (default: 0.05)
    :param policy_kwargs: Additional arguments for policy creation
    :param verbose: Verbosity level (0: none, 1: info, 2: debug)
    :param seed: Seed for pseudo-random generators
    :param device: Device to run on (cpu, cuda, auto)
    :param _init_setup_model: Whether to build the network on instantiation
    """

    def __init__(
        self,
        policy: Union[str, type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
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
        physics_type: str = 'none',  # 'none', 'energy_conservation', 'newtons_laws'
        lambda_phys: float = 0.1,
        mass: float = 1.0,
        gravity: float = 9.81,
        length: float = 1.0,
        dt: float = 0.05,
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
            # DDPG-specific settings (removing TD3 tricks)
            policy_delay=1,
            target_noise_clip=0.0,
            target_policy_noise=0.1,
            _init_setup_model=False,
        )

        # Physics-informed parameters
        self.physics_type = physics_type
        self.lambda_phys = lambda_phys
        self.mass = mass
        self.gravity = gravity
        self.length = length
        self.dt = dt

        # Use only one critic as in standard DDPG
        if "n_critics" not in self.policy_kwargs:
            self.policy_kwargs["n_critics"] = 1

        if _init_setup_model:
            self._setup_model()

    def _compute_physics_loss(self, observations: th.Tensor, actions: th.Tensor, next_observations: th.Tensor) -> th.Tensor:
        """
        Compute the physics-informed loss based on the selected type.

        Assumes specific state formats:
        - For pendulum (energy_conservation): state = [theta, theta_dot]
        - For robotics (newtons_laws): state = [position, velocity], action = force/torque

        :param observations: Current states (batch_size, obs_dim)
        :param actions: Actions (batch_size, act_dim)
        :param next_observations: Next states (batch_size, obs_dim)
        :return: Physics loss tensor (scalar)
        """
        if self.physics_type == 'energy_conservation':
            # For pendulum: energy = KE (kinetic energy/ Động Năng) + PE (Potential Energy/ Thế Năng)
            # KE = 0.5 * m * l^2 * theta_dot^2
            # PE = -m * g * l * cos(theta)
            theta, theta_dot = observations[:, 0], observations[:, 1]
            next_theta, next_theta_dot = next_observations[:, 0], next_observations[:, 1]

            energy = 0.5 * self.mass * self.length**2 * theta_dot**2 - self.mass * self.gravity * self.length * th.cos(theta)
            next_energy = 0.5 * self.mass * self.length**2 * next_theta_dot**2 - self.mass * self.gravity * self.length * th.cos(next_theta)

            # Penalize energy change (accounting for action work; here simplified as MSE on delta_energy)
            # Optionally add work = action * theta_dot * dt for torque input
            delta_energy = next_energy - energy
            physics_loss = (delta_energy ** 2).mean()

        elif self.physics_type == 'newtons_laws':
            # For robotics: F = m * a, where a = (v_next - v) / dt
            # Assume state = [pos, vel], action = force
            velocity = observations[:, -1]  # Last dim as velocity (adjust indices as needed)
            next_velocity = next_observations[:, -1]

            accel = (next_velocity - velocity) / self.dt
            predicted_accel = actions.squeeze() / self.mass  # Assume action is force (1D for simplicity)

            physics_loss = F.mse_loss(predicted_accel, accel)

        else:  # 'none' or fallback to smoothness
            actions_pi = self.actor(observations)
            actions_pi_next = self.actor(next_observations)
            physics_loss = F.mse_loss(actions_pi, actions_pi_next)

        return physics_loss

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Train the DDPG agent with a physics-informed loss added to the actor update.

        :param gradient_steps: Number of gradient updates to perform
        :param batch_size: Size of the minibatch sampled from the replay buffer
        """
        # Switch to train mode
        self.policy.set_training_mode(True)

        # Update learning rate
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        physics_losses = []

        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute target Q-values (no noise in DDPG)
                next_actions = self.actor_target(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Compute current Q-values
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Critic loss (standard MSE)
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Actor update (with physics-informed loss)
            # Standard actor loss
            actions_pi = self.actor(replay_data.observations)
            actor_loss = -self.critic.q1_forward(replay_data.observations, actions_pi).mean()

            # Physics-informed loss
            physics_loss = self._compute_physics_loss(replay_data.observations, actions_pi, replay_data.next_observations)
            physics_losses.append(physics_loss.item())

            # Combined actor loss
            total_actor_loss = actor_loss + self.lambda_phys * physics_loss
            actor_losses.append(total_actor_loss.item())

            # Optimize actor
            self.actor.optimizer.zero_grad()
            total_actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
            polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
            polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        # Logging
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/physics_loss", np.mean(physics_losses))
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
        )https://www.amazon.com/ref=nav_logo#