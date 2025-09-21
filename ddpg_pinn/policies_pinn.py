# DDPG can be view as a special case of TD3
from stable_baselines.td3.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, Actor, ContinuousCritic, TD3Policy  # noqa:F401
# Add explicit imports for base classes you'll customize
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.torch_layers import BaseFeaturesExtractor
from stable_baselines.common.preprocessing import get_action_dim
from gymnasium import spaces
from typing import Optional
import torch
import torch.nn as nn

# Step 1: Add KANLayer class here
class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Layer for PIKAN.
    Each edge applies a learnable B-spline activation to univariate inputs.
    Simplified from efficient-KAN: B-spline order 3, grid size 5.
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Grid: uniform [-1, 1] extended for splines
        self.grid = nn.Parameter(torch.linspace(-1, 1, grid_size + 2 * spline_order + 1).unsqueeze(0).unsqueeze(0), requires_grad=False)

        # Spline coefficients: [out, in, grid_size + spline_order]
        self.coefficients = nn.Parameter(torch.zeros(out_features, in_features, grid_size + spline_order))

        # Initialize with sine-based functions
        self.reset_parameters()

    def reset_parameters(self):
        # Sine initialization for activations
        with torch.no_grad():
            for i in range(self.out_features):
                for j in range(self.in_features):
                    self.coefficients[i, j] = torch.sin(torch.linspace(0, 2 * torch.pi, self.coefficients.shape[-1]))

    def b_spline_basis(self, x: torch.Tensor, grid: torch.Tensor, order: int) -> torch.Tensor:
        """
        Compute B-spline basis functions (recursive).
        x: [batch, in]
        Returns: [batch, in, grid_size + order]
        """
        x = x.unsqueeze(-1)  # [batch, in, 1]

        if order == 0:
            return ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()

        basis_lower = self.b_spline_basis(x, grid, order - 1)
        left = (x - grid[:, :-order]) / (grid[:, order:] - grid[:, :-order]) * basis_lower[:, :, :-1]
        right = (grid[:, order + 1 :] - x) / (grid[:, order + 1 :] - grid[:, 1:-(order - 1)]) * basis_lower[:, :, 1:]
        return left + right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_features]
        # Compute basis for each input dimension
        basis = self.b_spline_basis(x, self.grid, self.spline_order)  # [batch, in, basis_dim]

        # Apply coefficients: sum over basis per edge
        activations = torch.einsum('bid,oid->bo', basis, self.coefficients)  # [batch, out]

        return activations

# Step 2: Define custom KANActor (inherits from imported Actor)
class KANActor(Actor):
    """
    Custom Actor using PIKAN (KAN-based) for TD3/DDPG.
    Overrides the network to use KAN layers instead of MLP.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,  # Unused in KAN, kept for compatibility
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        # Build KAN layers instead of MLP
        kan_layers = []
        prev_dim = features_dim
        for hidden_dim in net_arch:
            kan_layers.append(KANLayer(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        kan_layers.append(KANLayer(prev_dim, action_dim))  # Final layer to action
        self.mu = nn.Sequential(*kan_layers)  # Deterministic output

# Step 3: Define custom KANContinuousCritic (inherits from imported ContinuousCritic) - Optional but recommended
class KANContinuousCritic(ContinuousCritic):
    """
    Custom Critic using PIKAN (KAN-based) for DDPG/TD3.
    Overrides the q_net to use KAN layers.
    """
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
            # Use KAN for q_net (input: features + action)
            q_layers = []
            prev_dim = features_dim + action_dim
            for hidden_dim in net_arch:
                q_layers.append(KANLayer(prev_dim, hidden_dim))
                prev_dim = hidden_dim
            q_layers.append(KANLayer(prev_dim, 1))  # Output Q-value
            q_net = nn.Sequential(*q_layers)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

# Custom TD3Policy using KANActor and KANContinuousCritic
class KANTD3Policy(TD3Policy):
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> KANActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return KANActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> KANContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return KANContinuousCritic(**critic_kwargs).to(self.device)

# Aliases for DDPG usage (replace MlpPolicy etc. with KAN versions if desired)
MlpPolicy = KANTD3Policy
CnnPolicy = KANTD3Policy  # Customize further if needed for CNN
MultiInputPolicy = KANTD3Policy