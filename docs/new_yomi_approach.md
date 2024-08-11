## Approximating Nash equilibria through play

The trouble with the previous approach was in finding good nash equilibrium solvers that can solve large state spaces (>10 cards) quickly.

Instead, we try to approximate something like a Nash equilibrium by letting pairs of models play against each other and optimising their strategy mutually and hope that the outcome is close enough to a true equilibrium state while taking into account the lack of complete information. This should be possible, see for example [this paper](https://openreview.net/pdf?id=cc8h3I3V4E). This means a player does not try to simulate the state of the opponents mind explicitly, but rather this information is captured by the model.

In this new approach, just like in the old one , we try to approximate the payout function by training a model to predict given a game state $s$ what the chance of a win is $X=1$,

$$P(X=1|s,\pi_{\mbox{self}} ,\pi_{\mbox{other}}) = f(s) \approx \bar{f}(s)$$

where $\pi_{\mbox{self}} ,\pi_{\mbox{other}}$ are the policies of both players, $\bar{f}(s)$ is the estimator we want to train and $s$ is the complete state containing all the knowledge of the player, such as identity of own cards, number of opponent cards, identity of known opponent cards, hp and status effects.
Such a model is straightforward to train if the opponent policy stays constant.

We will also need a policy network (though for the full yomi game more, to model the actions to take for other actions, such as exchanges, boosting combos and pumping play, and other non--cardplay actions).
Such a model should return a probability distribution over the available actions (card plays) $\mathcal{A}$

$$P(a| s, \pi_{\mbox{self}} ,\pi_{\mbox{other}}), a \in \mathcal{A} = g(a,s) = \bar{\mathbf{g}}(s),$$
where $\bar{\mathbf{g}}$ is a distribution over the action space.
How would such a model be trained?

As we play, we sample from the action distribution and could punish or reward the model based on the outcome of that particular interaction. That would however be wasteful, as the model would not have made use of the actions in would not have chosen.

Instead, we can map for each action taken and the actual action $z$ by the other player, how the state would have evolved

$$ \begin{array}{c} s + z + a_1 \rightarrow s_1  \\
s + z + a_2 \rightarrow s_2 \\
\vdots \\
s + z + a_n \rightarrow s_n
 \end{array}$$

While in principle we don't know whether which of these states is best, we can let the previous model decide. We could define a reward vector in such a way

$$\mathbf{f}: \mathbf{f}_i = \bar{f}(s_i) - \bar{f}(s),$$

or any other monotenous function of the difference, such as

$$\mathbf{f}: \mathbf{f}_i = \log \bar{f}(s_i) - \log \bar{f}(s),$$

One objective could be to maximise the expected reward

$$ \max  \bar{\mathbf{f}}(s)^T \bar{\mathbf{g}}(s) ,$$

or

$$L = - \bar{\mathbf{f}}(s)^T  \bar{\mathbf{g}}(s) ,$$

which can naturally be batched over different iteration situations $s$. Having 2 models here is actually a benefit, since the errors of the models are independent, and so the gradient should be unbiased.


The research in [this paper](https://openreview.net/pdf?id=cc8h3I3V4E) suggests that there is benefit, despite there being a different setup due to the unknown nature of the opponents action choices, to align the gradient with the tangent space to the simplex $\sum g_i(s) = 1$. If we have $m$ actions available, we can project using a projection operator

$$\Pi = \mathbf{I}_m - \frac{1}{m}\mathbf{1}_m \mathbf{1}_m^T.$$

Since $\Pi = \Pi^T = \Pi^2$ we can modify the loss to be any of

$$L = - (\Pi\bar{\mathbf{f}}(s))^T  \bar{\mathbf{g}}(s)  =  - \bar{\mathbf{f}}(s)^T  (\Pi\bar{\mathbf{g}}(s)) = - (\Pi\bar{\mathbf{f}}(s))^T (\Pi\bar{\mathbf{g}}(s)), $$

which would suggest that the gradient of the loss with respect to the parametrisation of $ \bar{\mathbf{g}}$ would be a projection as well and leave the simplex intact (though the model itself would naturally renormalise the outputs to sum to 1).
