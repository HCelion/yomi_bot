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

which would suggest that the gradient of the loss with respect to the parametrisation of $\bar{\mathbf{g}}$ would be a projection as well and leave the simplex intact (though the model itself would naturally renormalise the outputs to sum to 1).

## Regret based approach

While the above reasoning seems in principle sound, in practice the resulting equilibria are not stable, as small mismatches in the hyper parameters lead to over controlling and oscillating behaviour in the space of play.

Instead, we can try for the model to predict the regret per action taken

$$R(a) =  r(a) - r^*,$$
where $r^*$ is the reward under the best of the available actions and $r(a)$ is the, potentially counterfactual, reward we got for that action.
Normally these are added together per action and the strategy of a player updates as

$$P(a) = \frac{\max(R(a),0)}{\sum_{a'} \max(a,R(a'))}.$$


## WOLF learning

The problem with the raw regret and performance based update schemes is that the models tend to overshot in training and the strategies do not converge to the Nash equilibrium.
Instead, we will use the WOLF strategy, which stands for *Win Or Learn Fast*. The idea is that if to keep an average of the historic stategies in memory and compare them to the current strategy. If the current strategy performs better, the update step slows down, otherwise it grows. That change in dynamics is enough to force the Nash equilibrium.

In particular, there are two update speeds, one for winning $\delta_w$ and one for losing $\delta_l$, with $\delta_l > \delta_w$. Further there is an update parameter $\alpha$.
At any given state $s$, we try to measure the value of each available action $a\in \mathcal{A}.$

If we know the transition function $T(s,a) \rightarrow s'$, we can update for each action (even the unobserved ones, since we know the counterfactual transitions `s'_{a}` and the counterfactual rewards $r_a$).

We play according to a randomly instantiated policy $\pi(s,a)$

$$Q(s,a) = (1-\alpha)\,Q(s,a) + \alpha \left( r_a  + \gamma Q(s'_a)\right).$$

In addition, for each state $s$ we calculate the actually observed action $a_{obs}$

$$C(s, a_{obs}) += 1.$$

We can use these counts to keep track of our historical average strategy

$$\bar{\pi}(s,a)  =  \frac{C(s, a)}{\sum_a C(s,a)}$$

When we observe the other players reaction and the rewards and consequences of all the actions we could have played according to our strategy, we can try to establish which, $\pi(s,a)$ or $\bar{\pi}(s,a)$ is better, by averaging over the expected value in state $s$

$$V_{\pi}(s) = \sum_a \pi(s,a) Q(s,a).$$

If $V_{\pi}(s) > V_{\bar{\pi}}$, i.e. our current strategy is better than the historic average, we use $\delta = \delta_w$, otherwise we use $delta = \delta_l > \delta_w$. We update the action that maximises the new value function

$$a_{max} = \max_{a} Q(s,a)$$

Then

$$\pi_{s,a_{max}} = \pi_{s,a_{max}} + \delta$$
and all other actions get reduced in likelihood, whereas probability mass is taken equally from all non-maximising actions

$$\pi_{s,a} = \pi_{s,a} - \frac{\delta}{|A|-1}.$$

Also we have to clip and normalise to guarantee a correct probability distribution for $\pi(s,a)$, since updates can in principle push $\pi(s,a)$ outside the region $[0,1]$. Because we constantly compare to the average value of the long term strategy, when we are doing well, we lower the learning rate and stay closer to the true equilibrium value.

## WOLF in deeper models

If we want to roll out the Wolf strategy to a fuller problem, such as Yomi, we run into some problems. For once, the state space $\mathcal{S}$ can become very large and there is some structure in it, so it would be reasonable to expect that nearby states to $s$, for example with similar but not exactly the same cards and enemies, we can improve on learning.
Our goal is to fomulate the problem such that our state space dependence is handled by a deep model. All counts and states that are indexed by $s$ will now be handled by a model. These are $Q(s,a), \bar{\pi}(s,a)$ and $\pi(s,a).$
We now need to find loss functions for these such that we can train a model on it. As before we assume we know the rewards, which are trained functions $V$ of the value of the transition $s \rightarrow s'$.

For the average policy, we can just use the actually played actions as target for new pairs of $s,a$

$$L_{\mathbf{\bar{\pi}}} = \mbox{LogLoss}(\mathbf{\bar{\pi}}(s), \mathbf{1}_a)$$
where $\mathbf{1}_a$ is a vector that is $1$ for the observed action and zero otherwise.

For the action value function $Q(s,a)$ we can use the counterfactual loss

$$L_{Q(s,a)} = (Q(s,a)- (\mathbf{r} + \gamma V(\mathbf{s'})))^T\cdot(Q(s,a)- (\mathbf{r} + \gamma V(\mathbf{s'}))),$$

where $V(\mathbf{s'})$ is our model to predict the winningness of a state applied to all $s'$ that follow from $s+a'=s'$ over all available actions at $s$.
In particular for the Yomi game, we might actually want to set $\mathbf{r} = 0, \gamma=1$ and purely optimise for the value of the next game states, since the only true reward is winning or losing.

Lastly we need the loss for the actual policy. If the actual policy is in a losing state compared to the long term average, then the loss should be larger to accelerate policy movement according to the WoLF principle.

$$L_{\pi(s,a)} = \mbox{sigmoid}(\mathbf{Q(s)}\cdot \left(\mathbf{\bar{\pi}}(s) - \mathbf{\pi}(s)\right) ) \mbox{LogLoss}(\mathbf{\pi}(s),\mathbf{Q_{\mbox{max}}} ),$$


where $\mathbf{Q_{\mbox{max}}} = 1/|A_{max}|$ for all actions $A_{max}$ that maximise $\mathbf{Q}(s,a)$ and zero otherwise. The $\mathbf{\pi}(s)$ should not contribute to the gradient though, it is just accentuating the loss.
