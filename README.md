This repository represents the research work we have done in redesigning traditional neural networks into Physics-Informed neural networks (PINNs) and applying these systems in the financial engineering field. Some of our contributions consist of:

- Utilizing Newton's laws and Navier-Stokes Fluid Dynamic Theory to calculate the Physics Loss of the training process
- Experimenting the redesigned PINNs with 3 10-asset portfolios from three different financial markets: USA, Vietnam, and China.
- Construct a new PINNs structure called Physics-Informed Kolmogorov-Arnold Networks (KANs) to decompose multivariate functions into sums of learnable univariate functions (B-splines).

In this paper, we utilize the innovative power of physics-informed Kolmogorov-Arnold networks (PIKANs), along with the recalculation approach of Newton's laws into physics losses. The core analogy is inspired by Newton's second law of motion ($F = m * a$), adapted to a financial context for portfolio optimization. Here, actions (e.g., portfolio weights or trades for multiple assets) are interpreted as "forces" applied to the system. At the same time, a specific feature from the state observations (the 1-day return) serves as a proxy for "velocity" (rate of change in asset prices/returns). The loss enforces consistency between the observed acceleration (derived from changes in velocity over time) and the predicted acceleration (force divided by mass), using mean squared error (MSE) to penalize discrepancies.
This approach embeds a physics analogy into the reinforcement learning (RL) update: it encourages policies that produce actions respecting momentum-like dynamics in market returns, potentially leading to more stable and interpretable portfolios with better risk-adjusted performance (e.g., higher Sharpe ratios). The elements and formulas are consistent, though minor implementation details (e.g., tensor reshaping) vary slightly due to algorithm-specific batch handling (on-policy in PPO vs. off-policy in TD3/DDPG). The calculation is as follow:

Observed Acceleration ($a^*$):

$$
a^* = \frac{\Delta v}{\Delta t} = \frac{v_{t+1}-v_t}{\Delta t}
$$

Predicted Acceleration ($\hat{a}$):

$$
F = m\times\hat{a} \Rightarrow \hat{a} = \frac{F}{m}
$$

Physics Loss:

$$
L_{\text{phys}} = \text{MSE}(\hat{a}, a^*) = \frac{1}{N} \sum_{i=1}^{N} (a_{\text{predicted},i} - a_{\text{observed},i})^2
$$

where:

\begin{itemize}
    \item Force (F): Actions applied per asset. This force is modeled by the agent's actions, such as batch size or the number of assets.
    \item Mass (m): A hyperparameter represents "inertia" or resistance to change in asset dynamics.
    \item Velocity (v): Proxied by the 1-day return (\texttt{ret1}) feature, extracted from observations for each asset. This assumes observations are historical windows of market data: including open, high, low, close, volume of an asset.
    \item Acceleration ($\hat{a},a^*$): Rate of change in velocity, computed over a time step dt (default 0.05, analogous to a trading interval).
    \item N: Total elements (Batch size multiplies with number of assets).
\end{itemize}
