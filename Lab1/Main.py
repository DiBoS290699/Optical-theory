import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('axes.formatter', useoffset=False)


def func(x):
    return np.exp(x * 1j/10)


def kernel(x, eps, alpha=1.0):
    return np.sin(np.pi * alpha * (x - eps)) / (np.pi*(x - eps))


def Fs(xs, epss, alpha=1.0):
    h_x = xs[1] - xs[0]
    matrix_A = kernel(xs[None, :], epss[:, None], alpha)
    vector_f = func(xs)
    Fs = np.dot(matrix_A, vector_f) * h_x
    return Fs


def plots_f(xs):
    f = func(xs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title('График модуля исходного сигнала')
    axes[0].plot(xs, np.abs(f))
    axes[0].grid()
    axes[1].set_title('График аргумента исходного сигнала')
    axes[1].plot(xs, np.angle(f))
    axes[1].grid()
    plt.show()


def plots_F(xs, epss, alpha=1.0):
    F = Fs(xs, epss, alpha)
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    axes[0].set_title('График амплитуды выходного сигнала')
    axes[0].plot(xs, np.abs(F))
    axes[0].grid()

    axes[1].set_title('График фазы выходного сигнала')
    axes[1].plot(xs, np.angle(F))
    axes[1].grid()
    plt.show()


def plots_K(xs, epss, alpha=1.0):
    X, Y = np.meshgrid(xs, epss)
    K = kernel(X, Y, alpha)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, np.abs(K))
    plt.title('График амплитуды ядра')
    plt.show()
    plt.close(fig)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, np.angle(K))
    plt.title('График фазы ядра')
    plt.show()


a = -200
b = 200
n = 1000
p = -2
q = 2
m = 1000
alpha = 1
xs = np.linspace(a, b, n)
epss = np.linspace(p, q, m)
plots_f(xs)
plots_F(xs, epss, alpha)
plots_K(xs, epss, alpha)

