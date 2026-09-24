# mle_logistic.py
# playing with maximum likelihood for logistic regression

import numpy as np

np.random.seed(0)

# ---------- PART 0: make the data ----------

NUM_DATAPOINTS = 1000


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def generate_data(x):
    a = 3   # beta_1
    b = -5  # beta_0
    prob = sigmoid(a * x + b)
    return np.random.uniform() < prob


x_data = np.random.uniform(high=3, size=NUM_DATAPOINTS)
y_data = [generate_data(x) for x in x_data]


# ---------- PART 1: log likelihood with a for loop ----------

def log_likelihood_per_point(beta_0, beta_1, x, y):
    prob = sigmoid(beta_1 * x + beta_0)
    # y_data has numpy booleans in it, and "np.True_ is True" is False
    # so we must use "if y:" here, NOT "if y is True:"
    if y:
        return np.log(prob)
    else:
        return np.log(1 - prob)


def total_log_likelihood(beta_0, beta_1, xs, ys):
    total_ll = 0
    for (x, y) in zip(xs, ys):
        total_ll += log_likelihood_per_point(beta_0, beta_1, x, y)
    return total_ll


print("PART 1: log likelihood with a for loop")
print("-" * 50)

aishwarya_beta_0 = 3
aishwarya_beta_1 = -3
irhum_beta_0 = 4
irhum_beta_1 = -1
true_beta_0 = -5
true_beta_1 = 3

ll_aishwarya = total_log_likelihood(aishwarya_beta_0, aishwarya_beta_1, x_data, y_data)
ll_irhum = total_log_likelihood(irhum_beta_0, irhum_beta_1, x_data, y_data)
ll_true = total_log_likelihood(true_beta_0, true_beta_1, x_data, y_data)

print("Aishwarya's estimate (beta_0=3, beta_1=-3):  log likelihood =", ll_aishwarya)
print("Irhum's estimate     (beta_0=4, beta_1=-1):  log likelihood =", ll_irhum)
print("True values           (beta_0=-5, beta_1=3): log likelihood =", ll_true)
print()

if ll_aishwarya > ll_irhum:
    print("Aishwarya's estimate is better (higher log likelihood, closer to 0)")
else:
    print("Irhum's estimate is better (higher log likelihood, closer to 0)")


# ---------- PART 2: gradient with JAX ----------

print()
print("PART 2: gradient with JAX")
print("-" * 50)

import jax
import jax.numpy as jnp

# turn the list of numpy booleans into a plain 0/1 vector
y_vec = np.array([1.0 if y else 0.0 for y in y_data])
x_vec = np.array(x_data)


def total_log_likelihood_jax(betas):
    beta_0 = betas[0]
    beta_1 = betas[1]
    p = jax.nn.sigmoid(beta_1 * x_vec + beta_0)
    ll = y_vec * jnp.log(p) + (1 - y_vec) * jnp.log(1 - p)
    return jnp.sum(ll)


grad_fn = jax.grad(total_log_likelihood_jax)

aishwarya_betas = jnp.array([3.0, -3.0])
gradient = grad_fn(aishwarya_betas)

print("Gradient at Aishwarya's estimate [beta_0=3.0, beta_1=-3.0]:")
print("  d/d beta_0 =", gradient[0])
print("  d/d beta_1 =", gradient[1])


# ---------- CHECKS ----------

print()
print("CHECKS")
print("-" * 50)

# check 1: vectorized log likelihood should match the for-loop version
ll_jax_value = total_log_likelihood_jax(aishwarya_betas)
check1 = np.isclose(float(ll_jax_value), float(ll_aishwarya))
print("Check 1 - vectorized ll matches for-loop ll:")
print("  for-loop value:   ", float(ll_aishwarya))
print("  vectorized value: ", float(ll_jax_value))
print("  match:", check1)

# check 2: hand derived gradient with plain numpy, should match jax gradient
p_np = sigmoid(aishwarya_beta_1 * x_vec + aishwarya_beta_0)
grad_beta_0_hand = np.sum(y_vec - p_np)
grad_beta_1_hand = np.sum((y_vec - p_np) * x_vec)

check2_beta_0 = np.isclose(float(gradient[0]), grad_beta_0_hand)
check2_beta_1 = np.isclose(float(gradient[1]), grad_beta_1_hand)
print()
print("Check 2 - JAX gradient matches hand-derived formula:")
print("  hand d/d beta_0 =", grad_beta_0_hand, " jax d/d beta_0 =", float(gradient[0]), " match:", check2_beta_0)
print("  hand d/d beta_1 =", grad_beta_1_hand, " jax d/d beta_1 =", float(gradient[1]), " match:", check2_beta_1)

# check 3: both gradient components should be positive at aishwarya's estimate
check3_beta_0 = float(gradient[0]) > 0
check3_beta_1 = float(gradient[1]) > 0
print()
print("Check 3 - both gradient components positive at Aishwarya's estimate:")
print("  d/d beta_0 > 0:", check3_beta_0)
print("  d/d beta_1 > 0:", check3_beta_1)

all_checks_passed = check1 and check2_beta_0 and check2_beta_1 and check3_beta_0 and check3_beta_1
print()
print("ALL CHECKS PASSED:", all_checks_passed)
