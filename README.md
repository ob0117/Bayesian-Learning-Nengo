# Online Bayesian Learning in Nengo

In this project, I implement a biologically inspired, online Bayesian learning rule for spiking neurons using [Nengo](https://www.nengo.ai/). The rule is adapted from [_Synaptic Plasticity as Bayesian Inference_](https://www.nature.com/articles/s41593-021-00809-5) by Aitchison et al. (2021), and modified to work within the [_Neural Engineering Framework (NEF)_](https://compneuro.uwaterloo.ca/files/publications/stewart.2012d.pdf) to support population-level learning for higher-scale biological models!

> **TL;DR**: Synapses will adapt their weights based on how uncertain they are about the correct value of the weight. Synapses with higher uncertainty (variance) update more aggressively in response to errors. Implementing this in Nengo allows for real-time Bayesian-style learning with interconnected spiking neuron populations!

```python
from learning_rules import Bayesian
import nengo

# Define prior mean and variance
prior_mean = np.full((n_post, n_pre), 0.5)
prior_var = np.full((n_post, n_pre), 0.02)

# Use Bayesian rule in a Nengo connection
nengo.Connection(
    pre, post,
    function=lambda x: [0],
    solver=nengo.solvers.NoSolver(weights=True),
    learning_rule_type=Bayesian(prior_mean, prior_var),
)
```
For compatibility with Nengo:  
- _Python 3.10_
- _NumPy 1.26.4_

## Background

In neuroscience, the **Bayesian brain hypothesis** suggests that the brain constantly updates its internal beliefs about the world using probabilistic inference. In this view, perception, learning, and action are all about minimizing error between a biological agent's internal models (predictions) about the world vs. the actual sensory inputs it receives over time.

Bayesian learning principles are used extensively in areas like robotics, and artificial intelligence, and computational neuroscience to model decision-making under uncertainty. However, there seems to be a lack of literature regarding the implementation of online (real-time) Bayesian learning rules in spiking neuron models specifically, which was a big motivator for this project!

Aitchison et al. (2021) propose that synaptic weights represent probability distributions, and synapses adjust their learning rates based on uncertainty, where synapses with high variance update more rapidly in response to error. This principle, they argue, allows for adaptive learning that aligns with Bayesian inference and biological observations of learning.

While Aitchison et al. applied their learning rule to single-neuron models with direct feedback, this project brings the rule into a more realistic setting by implementing it in Nengo. Nengo supports **interconnected spiking neuron populations**, which can allow for exploring Bayesian learning at a higher scale, with more complex, biologically plausible models.

## Learning Rule

### Bayesian Learning Equations (Aitchison et al.)
Synaptic weights $w_i$ are modeled as random variables with mean $\mu_i$ and variance $\sigma_i^2$. The learning rule updates these parameters using error $\delta$ and presynaptic activity $x_i$:

$$
\Delta \mu_i \approx \frac{\sigma_i^2}{\sigma_\delta^2} \cdot x_i \cdot \delta - \frac{1}{\tau} (\mu_i - \mu_{\text{prior}})
$$

$$
\Delta \sigma_i^2 \approx -\frac{\sigma_i^4}{\sigma_\delta^2} \cdot x_i^2 + \frac{1}{\tau} (\sigma_{\text{prior}}^2 - \sigma_i^2)
$$

Where:
- $\delta$ is the error signal (target - output)
- $\sigma_\delta^2$ is baseline variance in the system
- $\mu_{\text{prior}}, \sigma_{\text{prior}}^2$ are prior beliefs over the weight
- $\tau$ is a time constant that controls the tendency to drift toward the priors by default

In essence, these equations are extensions of the classical **Delta rule**, which updates weights based on an error signal and learning rate. In this Bayesian version, rather than using a fixed learning rate, the update is scaled by a coefficient that represents how uncertain the synapse is about its weight. When _information is received_, i.e. presynaptic activity $x_i$ is high, the weight will update in correspondence with the level of uncertainty. In the absence of information, the values will naturally tend to drift toward the prior values, controlled by $\tau$.

---

### Translating to the NEF

To adapt these update rules to the Neural Engineering Framework, I worked off of the simple **NEF-Style Delta learning rule** for synaptic weights:

$$
\Delta w_{ij} = \eta \cdot g_i \cdot \langle e_i, \delta \rangle \cdot a_j(t)
$$

Where:
- $\eta$ is a learning rate
- $g_i$ are the gains of the postsynaptic neurons
- $a_j(t)$ are the firing rates (activities) of the presynaptic neurons
- $\langle e_i, \delta \rangle$ represents the error signal being projected onto the encoders of the postsynaptic neuron population. Neural populations under the NEF communicate information through encoding and decoding processes.

This can then be extended to the Bayesian form by replacing the fixed learning rate with the dynamic scaling factor based on synaptic uncertainty.

#### Final Implemented Equations in Nengo

The final equations operate in log-space, using:
- $m_i = \log \mu_i$ (log-mean)
- $s_i^2 = \log \sigma_i^2$ (log-variance)

Original log-space equation derivations are given in Aitchison et al, and the full update rules for the Nengo adaptation follow:

$$
\Delta m_i = \frac{s_i^2 \mu_i}{\sigma_\delta^2} \cdot g(\delta, e_i) \cdot a_j(t) - \frac{1}{\tau}(m_i - m_{\text{prior}})
$$

$$
\Delta s_i^2 = -\frac{s_i^4 \mu_i^2}{\sigma_\delta^2} \cdot a_j(t)^2 - \frac{2}{\tau}(s_i^2 - s_{\text{prior}}^2)
$$

Where $\mu_i = \exp(m_i)$ and $\sigma_i^2 = \exp(s_i^2)$ in practice

---

## Example Experiments

In Nengo, learning occurs by adjusting the synaptic weights between neural populations (ensembles) over time. These weights can work to approximate a function that transforms an input stimulus into a desired output. Weights are updated based on the error signal ($\delta$) between the actual and target outputs. Over time, the network becomes more accurate at representing specific functions through population-level interactions.

<img width="400" alt="image" src="https://github.com/user-attachments/assets/8ddd0158-522b-4dd0-9974-32bd0313ea3d" />

In this context, weights are represented as random variables with mean and variance values, and updated based on the adapted Bayesian learning rule.


| **Sine Function Learning** | **Ramp Function Learning** | **Population Size Impact** |
|----------------------------|-----------------------------|-----------------------------|
| A sine function is applied to an input stimulus, which is passed between two neuron ensembles. Both Nengo's [PES rule](https://www.nengo.ai/nengo-examples/loihi/learn-communication-channel.html) (Delta rule) and the custom Bayesian rule attempt to learn the transformation between neuron populations over time. | A ramp function is applied to an input stimulus. The learning rules attempt to learn the transformation over time. The Bayesian rule adapts more quickly to the abrupt value change in the ramp function, while the Delta rule shows smoother & slower adjustments. | Different ensembles of 1, 10, and 100 neurons learn a white noise signal with the Bayesian rule. Learning accuracy improves significantly with larger populations, confirming that larger ensembles represent input signals more accurately. |
| <img src="https://github.com/user-attachments/assets/6381f59f-995a-4f6b-9882-fe3ae449d2d1" width="300"/> | <img src="https://github.com/user-attachments/assets/164b64af-f059-4982-ade5-2d7e0238c904" width="300"/> | <img src="https://github.com/user-attachments/assets/0b3715bf-734d-46dd-b329-2aad5294e797" width="300"/> |
