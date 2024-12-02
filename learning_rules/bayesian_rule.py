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

Relevant source code: https://github.com/nengo/nengo/blob/30f711c26479a94e486ab1862a1400dce5b3ffa0/nengo/learning_rules.py
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
        Prior means; best estimate of target weight means.
    prior_variance : ndarray
        Prior variances; best estimate of target weight variances.
    initial_mean: ndarray
        Means for starting weight values.
    initial_variance : ndarray
        Variances for starting weight values; initial uncertainty.
    baseline_variance : float
        Baseline variance in the system, ex. from noise.
    tau_learning : float
        Time constant for synaptic learning. Dictates how quickly
        priors are forgotten - smaller means faster adaptation.
    pre_synapse : `.Synapse`, optional
        Synapse model used to filter the pre-synaptic activities.
    post_synapse : `.Synapse`, optional
        Synapse model used to filter the post-synaptic activities.
        If None, ``post_synapse`` will be the same as ``pre_synapse``.
    """

    modifies = "weights"
    probeable = ("mean", "variance", "delta_mean", 
                 "delta_variance", "pre_filtered", "post_filtered")

    prior_mean = NdarrayParam("prior_mean")
    prior_variance =  NdarrayParam("prior_variance")
    initial_mean = NdarrayParam("initial_mean")
    initial_variance =  NdarrayParam("initial_variance")
    
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
        default=5.0
    )

    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)

    def __init__(
            self, 
            prior_mean, 
            prior_variance,
            initial_mean,
            initial_variance,
            pre_synapse=Default,
            post_synapse=Default,
    ):
        super().__init__(size_in=1)
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.initial_mean = initial_mean
        self.initial_variance = initial_variance
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

class SimBayesian(Operator):
    """
    Run simulation, and perform weight updates for Bayesian learning rule.
    """

    def __init__(
        self,
        pre_filtered,
        error,
        weights,
        mean,
        variance,
        delta_mean,
        delta_variance,
        prior_mean,
        prior_variance,
        initial_variance,
        baseline_variance,
        tau_learning,
        tag=None
    ):
        super().__init__(tag=tag)
        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, error, mean, variance]
        self.updates = [weights, delta_mean, delta_variance]
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.initial_variance = initial_variance
        self.baseline_variance = baseline_variance
        self.tau_learning = tau_learning

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.reads[0]]
        error = signals[self.reads[1]]
        mean = signals[self.reads[2]]
        variance = signals[self.reads[3]]

        weights = signals[self.updates[0]]
        delta_mean = signals[self.updates[1]]
        delta_variance = signals[self.updates[2]]

        prior_mean = self.prior_mean
        prior_variance = self.prior_variance
        initial_variance = self.initial_variance
        baseline_variance = self.baseline_variance
        tau_learning = self.tau_learning

        def step_simbayesian():
            n_neurons = pre_filtered.shape[0]

            # error term based on Delta/PES rule for weights
            error_term = np.outer( error, pre_filtered )

            # regulatory variance term, from Aitchison et al.
            summation = np.sum(pre_filtered * dt * (1 - pre_filtered * dt))
            variance_reg = (prior_variance + initial_variance) * summation + baseline_variance

            # updates to mean and variance, adapted from Aitchison et al.
            delta_mean[...] = (variance * mean / variance_reg) * error_term \
                            - (1 / tau_learning) * (mean - prior_mean)
            delta_variance[...] = - (np.square(variance) * np.square(mean) / variance_reg) \
                            * np.square(pre_filtered) \
                            - (2 / tau_learning) * (variance - prior_variance)
            
            # update mean & variance
            mean[...] += delta_mean
            variance[...] += delta_variance

            np.maximum(variance, 1e-8, out=variance)
            # np.maximum(mean, 1e-8, out=mean)

            # sample new weights from normal distribution
            weights[...] = np.random.normal(mean, np.sqrt(variance), size=mean.shape)

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

    # signals for mean, variance, and their updates
    mean = Signal(bayesian.initial_mean, name="Bayesian:mean")
    variance = Signal(bayesian.initial_variance, name="Bayesian:variance")

    delta_mean = Signal(
        np.zeros_like(bayesian.initial_mean), 
        name="Bayesian:delta_mean"
    )
    delta_variance = Signal(
        np.zeros_like(bayesian.initial_variance), 
        name="Bayesian:delta_variance"
    )

    # reset updates to mean and variance so they don't accumulate
    model.add_op(Reset(delta_mean))
    model.add_op(Reset(delta_variance))

    # define an input error signal
    error = Signal(shape=rule.size_in, name="Bayesian:error")    
    model.sig[rule]["in"] = error

    # pre-synaptic activities
    pre_filtered = build_or_passthrough(
        model, bayesian.pre_synapse, model.sig[conn.pre_obj]["out"]
    )
    post_filtered = build_or_passthrough(
        model, bayesian.post_synapse, model.sig[get_post_ens(conn).neurons]["out"]
    )

    # get per-neuron error signal, by projecting it onto next population's encoders.
    # adapted from Nengo's PES implementation:
    # https://github.com/nengo/nengo/blob/30f711c26479a94e486ab1862a1400dce5b3ffa0/nengo/builder/learning_rules.py#L794

    if conn._to_neurons:
        # local_error = dot(encoders, error)
        post = get_post_ens(conn)
        encoders = model.sig[post]["encoders"]

        # TODO: currently assuming conection to neuron object
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
            mean, 
            variance,
            delta_mean,
            delta_variance,
            bayesian.prior_mean, 
            bayesian.prior_variance,
            bayesian.initial_variance, 
            bayesian.baseline_variance, 
            bayesian.tau_learning,
        )
    )

    # expose for probes
    model.sig[rule]["mean"] = mean
    model.sig[rule]["variance"] = variance
    model.sig[rule]["delta_mean"] = delta_mean
    model.sig[rule]["delta_variance"] = delta_variance
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered