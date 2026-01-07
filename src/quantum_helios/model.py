import equinox as eqx
import pennylane as qml
import jax.numpy as jnp
import jax.random as jr
import jax

class OutputLogRangeAdaptor(eqx.Module):
    """Hard-constraint adaptor using log-space mapping."""
    ylog_min: jnp.ndarray
    ylog_max: jnp.ndarray

    def __init__(self, ranges):
        y_min = jnp.array([r[0] for r in ranges], dtype=jnp.float64)
        y_max = jnp.array([r[1] for r in ranges], dtype=jnp.float64)
        self.ylog_min, self.ylog_max = jnp.log(y_min), jnp.log(y_max)

    def to_physical(self, y_raw):
        sig = jax.nn.sigmoid(y_raw)
        y_log = self.ylog_min + sig * (self.ylog_max - self.ylog_min)
        return jnp.exp(y_log)

class QuantumLayer(eqx.Module):
    params: jnp.ndarray
    n_wires: int
    _qnode: callable

    def __init__(self, n_wires, n_layers, key):
        self.n_wires = n_wires
        self.params = jr.normal(key, (n_layers, n_wires, 3)) * 0.2
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def qnode(x, w):
            qml.AngleEmbedding(x, wires=range(n_wires), rotation="Y")
            qml.StronglyEntanglingLayers(w, wires=range(n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]
        self._qnode = qnode

    def __call__(self, x):
        return jnp.stack(self._qnode(x, self.params)).astype(jnp.float32)

class PIQNN(eqx.Module):
    q_layer: QuantumLayer
    adaptor: OutputLogRangeAdaptor

    def __init__(self, in_dim, out_dim, n_layers, adaptor, key):
        self.q_layer = QuantumLayer(in_dim, n_layers, key)
        self.adaptor = adaptor

    def __call__(self, x):
        y_raw = self.q_layer(x)[:3] # Reconstruct 3 targets
        return self.adaptor.to_physical(y_raw)
