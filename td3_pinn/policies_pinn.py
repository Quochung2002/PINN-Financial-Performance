from typing import Any, Optional, Union
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Layer for PIKAN.
    Each edge applies a learnable B-spline activation to univariate inputs.
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid = nn.Parameter(th.linspace(-1, 1, grid_size + 2 * spline_order + 1).unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.coefficients = nn.Parameter(th.zeros(out_features, in_features, grid_size + spline_order))
        self.reset_parameters()

    def reset_parameters(self):
        with th.no_grad():
            nn.init.orthogonal_(self.coefficients, gain=1.0)

    def b_spline_basis(self, x: th.Tensor, grid: th.Tensor, order: int) -> th.Tensor:
        num_knots = grid.shape[-1]
        if order == 0:
            return ((x >= grid[..., 0 : num_knots - 1]) & (x < grid[..., 1 : num_knots])).float()
        basis_lower = self.b_spline_basis(x, grid, order - 1)
        left_numer = x - grid[..., 0 : num_knots - (order + 1)]
        left_denom = grid[..., order : num_knots - 1] - grid[..., 0 : num_knots - (order + 1)]
        left = left_numer / left_denom * basis_lower[..., 0 : basis_lower.shape[-1] - 1]
        right_numer = grid[..., order + 1 : ] - x
        right_denom = grid[..., order + 1 : ] - grid[..., 1 : num_knots - order]
        right = right_numer / right_denom * basis_lower[..., 1 : ]
        value = left + right
        return th.nan_to_num(value)

    def forward(self, x: th.Tensor) -> th.Tensor:
        grid = self.grid.expand(x.size(0), self.in_features, -1)
        x = x.unsqueeze(-1)  # Unsqueeze here once, outside recursion
        basis = self.b_spline_basis(x, grid, self.spline_order)
        activations = th.einsum('bfd,ofd->bo', basis, self.coefficients)
        return activations
    
    
class Actor(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        action_dim = get_action_dim(self.action_space)
        actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        self.mu = nn.Sequential(*actor_net)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return self.mu(features)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self(observation)

class KANActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        squash_output: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
            # Do NOT pass squash_output here, as the parent hardcodes it
        )
        action_dim = get_action_dim(self.action_space)
        kan_layers = []
        prev_dim = features_dim
        for hidden_dim in net_arch:
            kan_layers.append(KANLayer(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        kan_layers.append(KANLayer(prev_dim, action_dim))
        if squash_output:
            kan_layers.append(nn.Tanh())
        self.mu = nn.Sequential(*kan_layers)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                grid_size=self.mu[0].grid_size if self.mu else 5,
                spline_order=self.mu[0].spline_order if self.mu else 3,
            )
        )
        return data
    
class KANContinuousCritic(ContinuousCritic):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )
        action_dim = get_action_dim(self.action_space)
        self.q_networks = []
        for idx in range(n_critics):
            q_layers = []
            prev_dim = features_dim + action_dim
            for hidden_dim in net_arch:
                q_layers.append(KANLayer(prev_dim, hidden_dim))
                prev_dim = hidden_dim
            q_layers.append(KANLayer(prev_dim, 1))
            q_net = nn.Sequential(*q_layers)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                grid_size=self.q_networks[0][0].grid_size if self.q_networks else 5,
                spline_order=self.q_networks[0][0].spline_order if self.q_networks else 3,
            )
        )
        return data

class TD3Policy(BasePolicy):
    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = [256, 256]
            else:
                net_arch = [400, 300]
        actor_arch, critic_arch = get_actor_critic_arch(net_arch)
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )
        self.share_features_extractor = share_features_extractor
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode

class KANTD3Policy(TD3Policy):
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> KANActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return KANActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> KANContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return KANContinuousCritic(**critic_kwargs).to(self.device)

class CnnPolicy(TD3Policy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

class MultiInputPolicy(TD3Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

MlpPolicy = KANTD3Policy
CnnPolicy = KANTD3Policy
MultiInputPolicy = KANTD3Policy