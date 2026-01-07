import jax
jax.config.update("jax_enable_x64", True)
import equinox as eqx
import optax
import jax.random as jr
from quantum_helios.physics import physics_residuals
from quantum_helios.model import PIQNN, OutputLogRangeAdaptor

def main():
    # 1. Hyperparameters
    KEY = jr.PRNGKey(42)
    LR = 5e-3
    BATCH_SIZE = 4
    
    # 2. Inisialisasi Adaptor & Model
    ranges = [(1e-5, 6e-5), (2.0*6.9e8, 3.0*6.9e8), (3e4, 8e4)]
    adaptor = OutputLogRangeAdaptor(ranges)
    model = PIQNN(in_dim=8, out_dim=3, n_layers=5, adaptor=adaptor, key=KEY)
    
    # 3. Setup Optimizer
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    print("🚀 PIQNN Training Engine Started...")
    # Lanjutkan dengan training loop Anda...

if __name__ == "__main__":
    main()
