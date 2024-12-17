from nengo.learning_rules import LearningRuleType
from nengo.params import Default, NumberParam, NdarrayParam
from nengo.synapses import Lowpass, SynapseParam
from nengo.ensemble import Ensemble
from nengo.node import Node
from nengo.builder import Builder
from nengo.builder.signal import Signal
from nengo.builder.operator import Reset, Operator, DotInc
import numpy as np

"""
Define a new Bayesian learning rule.

Adapted from: https://github.com/nengo/nengo/blob/30f711c26479a94e486ab1862a1400dce5b3ffa0/nengo/learning_rules.py
"""

def _remove_default_post_synapse(argreprs, default):
    default_post_synapse = f"post_synapse={default!r}"
    if default_post_synapse in argreprs:
        argreprs.remove(default_post_synapse)
    return argreprs

class Bayesian(LearningRuleType):
    r"""
    Bayesian Learning Rule.

    Source: https://www.nature.com/articles/s41593-021-00809-5 
    (2021, Aitchison et al.)

    Modifies connection weights, modeled as normal distributions, each
    having their own mean and variance.
    
    
    Notes
    -----
    This linear Bayesian learning rule is adapted from Aitchison et al.
    It can be interpreted as an extension of the Delta rule, where the
    variance over time sets the learning rate instead of some fixed
    value.

    Weights are updated based on an error signal, current mean and 
    variance, prior (target) mean and variance, and initial variance in 
    the system. Prior mean and variance are given by a set of target 
    weights.

    For each weight, the mean changes according to the variance. The 
    higher the uncertainty (or greater the variance), the larger the 
    change in the mean will be. When firing rates are low (in the absence 
    of information), the inferred mean will move toward the prior.
    
    The change in variance, derived by Aitchison et al., is intended to 
    reduce uncertainty when firing rates are high, and at the same time, 
    continually increase uncertainty up to the prior (target) variance.

    
    Parameters
    ----------
    prior_mean : ndarray
        Prior means; initial estimate of target weight means.
    prior_variance : ndarray
        Prior variances; initial estimate of target weight variances.
    tau_learning : float
        Time constant for synaptic learning. Dictates how aggresively
        mean and variance drift toward priors.
    baseline_variance : float
        Baseline variance in the system, ex. from noise.
    pre_synapse : `.Synapse`, optional
        Synapse model used to filter the pre-synaptic activities.
    post_synapse : `.Synapse`, optional
        Synapse model used to filter the post-synaptic activities.
        If None, ``post_synapse`` will be the same as ``pre_synapse``.
    """

    modifies = "weights"
    probeable = ("mu", "sigma2", "pre_filtered", "post_filtered")

    prior_mean = NdarrayParam("prior_mean")
    prior_variance =  NdarrayParam("prior_variance")
    
    baseline_variance = NumberParam(
        "baseline_variance", 
        low=0, 
        low_open=True, 
        readonly=True, 
        default=0.02
    )
    tau_learning = NumberParam(
        "tau_learning", 
        low_open=True, 
        high_open=True, 
        readonly=True, 
        default=1000
    )

    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)

    def __init__(
            self, 
            prior_mean, 
            prior_variance,
            pre_synapse=Default,
            post_synapse=Default,
    ):
        super().__init__(size_in=1)
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )

    @property
    def _argreprs(self):
        return _remove_default_post_synapse(super()._argreprs, self.pre_synapse)
    

"""
Build the Bayesian learning rule model.

SimBayesian: responsible for executing the simulation, updating signals at each time step.
build_bayesian: make the Bayesian rule useable by Nengo. 

Relevant source code: https://github.com/nengo/nengo/blob/30f711c26479a94e486ab1862a1400dce5b3ffa0/nengo/builder/learning_rules.py
"""

def get_post_ens(conn):
    """Get the output `.Ensemble` for connection."""
    return (
        conn.post_obj
        if isinstance(conn.post_obj, (Ensemble, Node))
        else conn.post_obj.ensemble
    )

def build_or_passthrough(model, obj, signal):
    """Builds the obj on signal, or returns the signal if obj is None."""
    return signal if obj is None else model.build(obj, signal)

# convert mean & variance from a distribution in log space to linear space
def convert_log_to_linear(m, s2):
    mu = np.exp(m + s2 / 2)
    sigma2 = mu**2 * (np.exp(s2) - 1)
    
    return mu, sigma2

# convert mean & variance from a distribution in linear space to log space
def convert_linear_to_log(mu, sigma2):
    s2 = np.log( (sigma2 / mu**2) + 1 )
    m = np.log(mu) - s2 / 2
    
    return m, s2

# helper to avoid extreme values in the log space
def clip(value, name, epsi=1e-8, min_thresh=-10.0, max_thresh=10.0):
    if(name == "s2"):
        return np.clip(value, epsi, max_thresh)
    else:
        return np.clip(value, min_thresh, max_thresh)

class SimBayesian(Operator):
    """
    Run simulation, and perform weight updates for Bayesian learning rule.
    """

    def __init__(
        self,
        pre_filtered,
        error,
        weights,
        mu,
        sigma2,
        m,
        s2,
        delta_m,
        delta_s2,
        sigma2_prior,
        s2_prior,
        m_prior,
        sigma2_baseline,
        tau_learning,
        tag=None
    ):
        super().__init__(tag=tag)
        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, error]
        self.updates = [weights, delta_m, delta_s2, m, s2, mu, sigma2]
        self.sigma2_prior = sigma2_prior
        self.s2_prior = s2_prior
        self.m_prior = m_prior
        self.sigma2_baseline = sigma2_baseline
        self.tau_learning = tau_learning

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.reads[0]]
        error = signals[self.reads[1]]

        weights = signals[self.updates[0]]

        delta_m = signals[self.updates[1]]
        delta_s2 = signals[self.updates[2]]

        m = signals[self.updates[3]]
        s2 = signals[self.updates[4]]

        mu = signals[self.updates[5]]
        sigma2 = signals[self.updates[6]]

        sigma2_prior = self.sigma2_prior

        s2_prior = self.s2_prior
        m_prior = self.m_prior

        sigma2_baseline = self.sigma2_baseline
        tau_learning = self.tau_learning

        def step_simbayesian():
            # error term for weights based on Delta (PES) rule
            error_term = np.outer( error, dt * pre_filtered )

            # regulatory variance term in linear space, from Aitchison et al.
            sigma2_del = 2 * sigma2_prior * np.sum(pre_filtered * dt * (1 - pre_filtered * dt)) \
                        + sigma2_baseline
            
            # updates to log mean and variance, from Aitchison et al.
            delta_m[...] = clip(
                ( (s2 * mu) / sigma2_del ) * error_term \
                    - 1 / tau_learning * (m - m_prior),
                "delta_m"
            )
            delta_s2[...] = clip(
                 - ( (s2**2 * mu**2 * (dt * pre_filtered)**2) / sigma2_del ) \
                    - 2 / tau_learning * (s2 - s2_prior),
                "delta_s2"
            )

            m[...] = clip(
                m + delta_m, 
                "m"
            )
            s2[...] = clip(
                s2 + delta_s2, 
                "s2"
            )

            # store for probing
            mu[...], sigma2[...] = convert_log_to_linear(m, s2)

            # sample weights from normal distribution using current log parameters
            weights[...] = np.exp(m + np.sqrt(s2) * np.random.normal(0, 1, size=m.shape))

        return step_simbayesian

@Builder.register(Bayesian)
def build_bayesian(model, bayesian, rule):
    """
    Builds a `.Bayesian` object into a model.

    Parameters
    ----------
    model : Model
        The model to build into.
    bayesian : Bayesian
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.
    """

    conn = rule.connection

    m_prior, s2_prior = convert_linear_to_log(
        bayesian.prior_mean, 
        bayesian.prior_variance
    )

    # signals for mean and variance for probes
    mu = Signal(bayesian.prior_mean, name="Bayesian:mu")
    sigma2 = Signal(bayesian.prior_variance, name="Bayesian:sigma2")

    # updates to mean and variance happen in log space
    m = Signal(m_prior, name="Bayesian:m")
    s2 = Signal(s2_prior, name="Bayesian:s2")

    delta_m = Signal(
        np.zeros_like(bayesian.prior_mean), 
        name="Bayesian:delta_m"
    )
    delta_s2 = Signal(
        np.zeros_like(bayesian.prior_variance), 
        name="Bayesian:delta_s2"
    )

    model.add_op(Reset(delta_m))
    model.add_op(Reset(delta_s2))

    # define input error signal
    error = Signal(shape=rule.size_in, name="Bayesian:error")    
    model.sig[rule]["in"] = error

    # pre-synaptic activities
    pre_filtered = build_or_passthrough(
        model, bayesian.pre_synapse, model.sig[conn.pre_obj]["out"]
    )
    post_filtered = build_or_passthrough(
        model, bayesian.post_synapse, model.sig[get_post_ens(conn).neurons]["out"]
    )

    # get per-neuron error signal by projecting it onto next population's encoders.
    if conn._to_neurons:
        # local error = dot(encoders, error)
        post = get_post_ens(conn)
        encoders = model.sig[post]["encoders"]

        # NOTE: currently assuming connection to a neuron object only
        encoders = (encoders[:, conn.post_slice])

        local_error = Signal(shape=(encoders.shape[0],))
        model.add_op(Reset(local_error))
        model.add_op(DotInc(encoders, error, local_error, tag="Bayesian:encode"))
    else:
        local_error = error


    model.add_op(
        SimBayesian(
            pre_filtered, 
            local_error, 
            model.sig[conn]["weights"],
            mu,
            sigma2,
            m, 
            s2,
            delta_m,
            delta_s2,
            bayesian.prior_variance,
            s2_prior,
            m_prior,
            bayesian.baseline_variance, 
            bayesian.tau_learning,
        )
    )

    # expose values for probes
    model.sig[rule]["mu"] = mu
    model.sig[rule]["sigma2"] = sigma2
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered