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

def predict(w, phi):
    return np.matmul(w, phi.transpose())


# Generate sample data
x, t = gen_sine(n=10)
# Compute polynomials
w0, phi0 = fit_polynomial(x, t, M=0)
w2, phi2 = fit_polynomial(x, t, M=2)
w4, phi4 = fit_polynomial(x, t, M=4)
w8, phi8 = fit_polynomial(x, t, M=8)

x_pretty = np.linspace(0, 2 * np.pi, num=100, endpoint=True)

ax1 = plt.subplot(221)
ax1.plot(x_pretty, np.sin(x_pretty), color="green")
ax1.plot(x, t, "ro", color="blue")
ax1.plot(x, predict(w0, phi0), color="red")
ax1.text(5, 0.5, "M=0")
plt.ylabel("t")
plt.xlabel("x")

ax1 = plt.subplot(222)
ax1.plot(x_pretty, np.sin(x_pretty), color="green")
ax1.plot(x, t, "ro", color="blue")
ax1.plot(x, predict(w2, phi2), color="red")
ax1.text(5, 0.5, "M=2")
plt.ylabel("t")
plt.xlabel("x")

ax1 = plt.subplot(223)
ax1.plot(x_pretty, np.sin(x_pretty), color="green")
ax1.plot(x, t, "ro", color="blue")
ax1.plot(x, predict(w4, phi4), color="red")
ax1.text(5, 0.5, "M=4")
plt.ylabel("t")
plt.xlabel("x")

ax1 = plt.subplot(224)
ax1.plot(x_pretty, np.sin(x_pretty), color="green")
ax1.plot(x, t, "ro", color="blue")
ax1.plot(x, predict(w8, phi8), color="red")
ax1.text(5, 0.5, "M=8")
plt.ylabel("t")
plt.xlabel("x")

plt.show()
