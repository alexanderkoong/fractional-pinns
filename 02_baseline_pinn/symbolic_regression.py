#!/usr/bin/env python3
"""
Symbolic Regression for f(c) - Extract interpretable equation from trained PINN
"""
import os
import pickle
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pysr import PySRRegressor

# 1. Load trained f_params
with open('results/params.pkl', 'rb') as f:
    params = pickle.load(f)
c_params, f_params, gamma_params, beta_params = params

# 2. Define f(c) evaluation (from train_virality.py)
def fnn_fixed_boundary(params, x):
    boundary_term = x * (1 - x)
    for i in range(len(params) - 1):
        W, b = params[i]
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    x = jnp.exp(x @ W + b) * boundary_term
    return x

# 3. Sample f(c) at many points
c_vals = np.linspace(0, 1, 500)
c_input = jnp.array(c_vals.reshape(1, 500, 1), dtype=jnp.float32)
f_vals = np.array(fnn_fixed_boundary(f_params, c_input)[0, :, 0])

# 4. Add boundary enforcement (from existing symbolic.py pattern)
# Weight boundary points 10x to enforce f(0)=0, f(1)=0
c_with_bounds = np.concatenate([c_vals, [0.0, 1.0]])
f_with_bounds = np.concatenate([f_vals, [0.0, 0.0]])
weights = np.concatenate([np.ones(500), [10.0, 10.0]])

# 5. Run PySR symbolic regression (config from existing symbolic.py)
os.makedirs('models', exist_ok=True)
os.makedirs('expressions', exist_ok=True)

model = PySRRegressor(
    model_selection="best",
    niterations=300,
    maxsize=14,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp"],
    complexity_of_operators={"/": 2, "exp": 3},
    elementwise_loss="loss(x, y, w) = w*(x - y)^2",
    temp_equation_file="models/virality_f.csv",
    progress=True,
)
model.fit(c_with_bounds.reshape(-1, 1), f_with_bounds, weights=weights)

# 6. Display results
print("\n" + "="*60)
print("SYMBOLIC REGRESSION RESULTS")
print("="*60)
print(f"\nBest equation: f(c) = {model.sympy()}")
print(f"Complexity: {model.get_best()['complexity']}")

# 7. Plot comparison
c_plot = np.linspace(0, 1, 200)
f_nn = np.array(fnn_fixed_boundary(f_params, jnp.array(c_plot.reshape(1,200,1)))[0,:,0])
f_symbolic = model.predict(c_plot.reshape(-1, 1))

correlation = np.corrcoef(f_nn, f_symbolic)[0, 1]
mse = np.mean((f_nn - f_symbolic)**2)

plt.figure(figsize=(10, 6))
plt.plot(c_plot, f_nn, 'b-', linewidth=2, label='Neural Network f(c)')
plt.plot(c_plot, f_symbolic, 'r--', linewidth=2, label=f'Symbolic: {model.sympy()}')
plt.xlabel('c (normalized engagement)', fontsize=12)
plt.ylabel('f(c)', fontsize=12)
plt.title(f'Learned vs Symbolic f(c)\nCorrelation: {correlation:.4f}, MSE: {mse:.6f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/symbolic_f_comparison.png', dpi=150)
plt.close()

# 8. Save results (pattern from existing symbolic.py)
with open('expressions/virality_f.txt', 'w') as f:
    f.write(str(model.sympy()) + '\n')

with open('results/symbolic_results.txt', 'w') as f:
    f.write("SYMBOLIC REGRESSION RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Best equation: f(c) = {model.sympy()}\n")
    f.write(f"Complexity: {model.get_best()['complexity']}\n")
    f.write(f"Correlation with NN: {correlation:.4f}\n")
    f.write(f"MSE: {mse:.6f}\n")
    f.write(f"\nAll candidates:\n")
    f.write(str(model.equations_))

print(f"\nCorrelation: {correlation:.4f}")
print(f"MSE: {mse:.6f}")
print("\nSaved: results/symbolic_f_comparison.png")
print("Saved: results/symbolic_results.txt")
print("Saved: expressions/virality_f.txt")
