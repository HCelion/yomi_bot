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

For derivations, look [here](https://www.cs.cmu.edu/~mmv/papers/02aij-mike.pdf)

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

An alternative approach for the loss is to get rid of the estimation of $Q(s,a)$ altogher  and use the counterfactual payoffs $\mathbf{r}$ as target

$$L_{\pi(s,a)} = \mbox{sigmoid}(\mathbf{r}\cdot \left(\mathbf{\bar{\pi}}(s) - \mathbf{\pi}(s)\right)) \, \mathbf{r}\cdot \mathbf{\pi}(s).$$

The sigmoid term can be replaced with any other function that assigns higher learnign rate to losing configurations, for example the the step function between $\delta_{min}$ and $\delta_{max}$ as in the non-gradient version.

## LOLA approach

For derivations, look [here](https://arxiv.org/pdf/1709.04326)

## Baselined policy learning

One way to train a policy model $\pi_theta(a|s)_\theta$ is to use the actor-critic approach.
We train a separate value model $V_{\phi}(s)$ separately to improve on the loss.

We let $M$ episodes run until each finished at time $T$. For each data point we can introduce the time horizon $n$, the discount factor $\gamma$.
The for each episode in $M$ we simulate the trace
$$\tau = \{s_0, a_0, r_0, s_1, \dots, s_T\}.$$
For each state alongside the trace we approximate the bootstrapped value

$$\hat{Q}_n(s_t, a_t) = \sum_{k=0}^{n-1}\gamma^k\cdot r_{t+k} + \gamma^n V_{\phi}(s_{t+n})$$

and the bootstrapped advantage

$$\hat{A}_n(s_t, a_t) = \hat{Q}_n(s_t, a_t) - V(s_t)$$


We can then update the networks with the gradients

$$ \begin{split}
\phi \rightarrow \phi - \alpha \nabla_{\phi} \sum_t (\hat{Q}_n(s_t, a_t) - V(s_t))^2 \\
\theta_t \rightarrow \theta - \alpha \nabla_{\theta}  \sum_t \left[\hat{A}_n(s_t, a_t)\log \phi_\theta(a_t|s_t)\right]
\end{split}
$$

Of course, for rock paper scissors it makes sense to discount completely $\gamma=1$ such that

$$\hat{Q}_n(s_t, a_t) = \hat{Q}(s_t, a_t) = r_t$$


## Deep WolF

Applying Wolf naively to a deep setting did not bring the results that were wished. However, the principal approach is sensible.
The paper [here](https://arxiv.org/pdf/1603.01121) has an approach based on the observation that strategies $\sigma$ can be mixed

$$\sigma_3 = \lambda_1 \sigma_1 + \lambda_2 \sigma_2$$

with $\lambda_1 + \lambda_2 = 1$. Thus a policy could first decide via a random throw of the die which strategy to follow in the next episode, and then run it. In the above paper $\sigma_1$ was tracking the historical average policy $\Pi,$ whereas $\sigma_2$ followed an Q-action model $\epsilon$-greedily.

One drawback of that method is that the mixing does not happen on the model level, but instead on a policy level. The greedy function returns pure predictions most of the time.

To get around this, one observation is that in reinforcement settings there is a model-data equivalence in the limit of large data being generated in each play through. Models are trained on past data, create new data through self play, and the next generation of models is trained on that new data.

So rather than mixing on the policy stage, we can mix data generated by different processes together in different ratios for the next generation of model being trained.

Let $\Pi_i(\theta)$ be a historical average model of player $i$. Let $\pi_i$ and alternative model. Let us for the moment assume that the state $s$ of the system is static, i.e. it is a pure matrix game.
Let $H_1$ be a history of self-play of the old model given by the list of tuples $(s, r, a, s')$ in the standard notation. From that history we can extract a approximate history $\hat{\Pi}(s)$. Similarly, we can generate an approximate history $\hat{\pi}(s)$ from $H_2$ as generated by $\pi$.
Clearly, if we generate $N_1$ sample of $\Pi$ and $N_2$ sample of $\pi$, the generated sample will have an effective strategy

$$\sigma(s) = \frac{N_1}{N_1+N_2}\Pi(s) + \frac{N_2}{N_1+N_2}\pi(s).$$

And an average policy model trained on that sample should follow $\sigma(s)$ approximately.

Rewriting this as

$$ \sigma \approx \Pi + \delta \pi,$$
where $\delta = \frac{N_2}{N_1}$. The sample mixing ratio approximates a learning rate of a policy.

In WoLF learning, we keep track of an average strategy and the current best strategy. If the current best strategy is losing compared to the average strategy, the learning rate $\delta$ is increased to $\delta_L$ and otherwise is reduced to $\delta_W < \delta_L$.

Similarly in deep Wolf, we have a current best model $\pi$ and an average model $\Pi$. We generate $N_{update}$ samples of reinforcement and train an updated policy $\pi_{\mbox{update}}$. We also run $N_{eval} = N_{update}$ runs against the average policy model $\Pi$. We can then compare the utilities of $\pi$ and $\Pi$ on $N_{update}$ and $N_{eval}$ respectively. If the current model is better than $\Pi$, we can create a smaller $N_w$ using $\pi_{\mbox{update}}$., if we are losing, we take a larger sample $N_w$.
We then mix the samples of $N_{update}$ and $N_{w/l}$, and train a new model $\pi'$ based off that. $N_{update}$ can be added to a large buffer to then update $\Pi'$.
this gives the effective $\delta_{l/w} = N_{l,w}/N_{\mbox{update}}$.

$$
\begin{array}{l}
\textbf{Hyperparameters:} N_{update}, N_L > N_W \text{ as integers} \\
\text{For each player } i :\\
\quad \text{Initialize } \pi_\theta(s) \text{ best policy arbitrarily} \\
\quad \text{Initialize } \Pi_{\theta'}(s) \text{ average policy arbitrarily} \\
\quad \text{Initialize } V_{\theta''}(s) \text{ valuation network} \\
\quad \text{Initialize } L \text{ circular long term buffer} \\
\quad \text{Initialize } S \text{ short term buffer} \\
\text{Until convergence:}\\
\text{Empty } S \\
\text{For } i = 1, N_{update}\text{ :} \\
\quad \text{Initialize state } s_{0} \\
\quad \textbf{while }   s_t \text{ not terminal:} \\
\quad \quad \text{Choose action } a \text{ from state } s_t \text{ using policy derived from } \pi_{\theta}(s_t) \\
\quad \quad \text{Take action } a, \text{ observe reward } r \text{ and next state } s_{t+1} \\
\quad \quad \text{ store } s,a, s_{t+1} \text{ in } L \\
\quad \quad \text{ store } s,a \text{ in } S \\
\quad \quad \textbf{end while}\\
\quad \text{Calculate bootstrapped rewards } \hat{Q}_n(s_t, a_t) = \sum_{k=0}^{n-1}\gamma^k\cdot r_{t+k} + \gamma^n V_{\theta''}(s_{t+n}) \\
\quad \text{Calculate Advantages } A(s_t) =\hat{Q}_n(s_t, a_t) - V(s_t) \\
\text{Train } \pi' \text{ by minimising }  L= -\sum  A(s_t) \log p'(a|s_t) \\
\text{Calculate mean utility } u_{\pi} \text{ from the } N_{update} \text{ samples } \\
\text{Play } N_{update} \text{ episodes using }\Pi \text{ and calculate utility } u_{\Pi} \\
\text{If } u_\pi > u_{\Pi} \text{ set } N_{next} = N_{W} \text{ otherwise } N_{L}\\
\text{Generate } N_{next} \text{ plays with policy } \pi' \text{ storing each observation } s,a \text{ in } S \\
\text{Train } \pi'' \text{ on samples from } S \text{ to minimize } L = - \sum \mathbb{I}_{a_t} \log\pi''(a_t|s_t) \\
\text{Set } \pi \leftarrow \pi''  \\
\text{Retrain } \Pi \text{ on samples from } L
\end{array}
$$

## A working approach

The regret based approach laid out in these papers worked well
[The OG CRM](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf)
[MC CRM](https://proceedings.neurips.cc/paper_files/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf)
[Deep CRM](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf)

## The value of nested Penny matching

A useful testbed for solving stochastic games is the nested Penny Matching game. Instead of playing a single Penny matching game, another game is being played based on the behaviour of the players in the first game.
There are two flavours to this. The game always starts for the first round in state $1$.
Depending on another condition the state of the second round is either $2$ or $3$.
There are essentially two pure versions:

Scenario 1: Player 1's action determines the state of which game is being played, say $2$ for $'Even'$ and $3$ for $'Odd'$
Scenario 2: Whether Player 1 wins decides whether the game moves to state $2$ or $3$.

Let $V_{2/3}$ be player $1$s value in game $2$ and $3$ respectively. The payoff matrices for the game in state $1$
are for the first

$$
P^1_1 = \left(\begin{array}{cc}
P_{11} + \gamma V_2 &  P_{12} + \gamma V_2 \\
P_{21} + \gamma V_3 &  P_{22} + \gamma V_3
\end{array}
\right)
$$

and for player $2$ it would be $P^1_2 = -P^1_1$.

In the second scenario where for sake of simplicity we assume that $'Even'/'Even'$ and $'Odd'/'Odd'$ are the winning states for $1$ we have

$$
P^2_1 = \left(\begin{array}{cc}
P_{11} + \gamma V_2 &  P_{12} + \gamma V_3 \\
P_{21} + \gamma V_3 &  P_{22} + \gamma V_2
\end{array}
\right)
$$

The values can be calculated at any point in time and the expected values of the games, as well as the regrets can be explicitly calculated. We can generally set $\gamma$ to $1$.
