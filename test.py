import numpy as np
import matplotlib.pyplot as plt


def gen_sine(n):
    # YOUR CODE HERE
    sigma = 0.25
    x = np.zeros(n)
    t = np.zeros(n)
    pi = np.pi
    nr = (2 * pi) / n
    x = np.linspace(0, 2 * np.pi, num=n, endpoint=True)
    mu = np.sin(x)
    t = np.random.normal(mu, sigma)

    return x, t


def designmatrix(x, M):  # it is highly recommended to write a helper function that computes Phi
    x_big = np.repeat(x, M + 1, axis=0).reshape(x.shape[0], M + 1)
    m_array = np.repeat(np.arange(M + 1), x.shape[0]).reshape(M + 1, x.shape[0]).transpose()
    Phi = np.power(x_big, m_array)
    return Phi


def fit_polynomial(x, t, M):
    Phi = designmatrix(x, M)
    inv = np.linalg.inv(np.matmul(Phi.transpose(), Phi))
    w_ml = np.matmul(np.matmul(inv, Phi.transpose()), t)
    return w_ml, Phi


## Test your function
np.random.seed(42)
N = 10
x, t = gen_sine(N)

assert x.shape == (N,), "the shape of x is incorrect"
assert t.shape == (N,), "the shape of t is incorrect"

### Test your function
N = 10
x = np.square((np.linspace(-1, 1, N)))
t = 0.3 * x + 2.5
m = 2
w, Phi = fit_polynomial(x, t, m)

assert w.shape == (m + 1,), "The shape of w is incorrect"
assert Phi.shape == (N, m + 1), "The shape of Phi is incorrect"

print("test")

# 1.3 YOUR CODE HERE
# Generate sample data
x, t = gen_sine(n=10)
# Compute polynomials
p0 = fit_polynomial(x, t, M=0)
p2 = fit_polynomial(x, t, M=2)
p4 = fit_polynomial(x, t, M=4)
p8 = fit_polynomial(x, t, M=8)

ax1 = plt.subplot(2, 2, 1)
ax1.plot(x,t)
ax1.plot(x,p0)
plt.show()
