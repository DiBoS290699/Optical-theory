import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return np.exp(x * 1j/10)


def kernel(x, eps, alpha=1.0):
    return np.sin(np.pi * alpha * (x - eps)) / (np.pi*(x - eps))


def Fs(xs, epss, alpha=1.0):
    h_x = xs[1] - xs[0]
    matrix_A = []
    for eps in epss:
        matrix_A.append([kernel(xs, eps, alpha)])
    vector_f = func(xs)
    Fs = np.dot(matrix_A, np.transpose(vector_f)) * h_x
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title('График модуля результата преобразования')
    axes[0].plot(xs, np.abs(F))
    axes[0].grid()
    axes[1].set_title('График аргумента результата преобразования')
    axes[1].plot(xs, np.angle(F))
    axes[1].grid()
    plt.show()


# def plots_K(xs, epss, alpha=1):
#     K = kernel(xs, epss, alpha)
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     axes[0].set_title('График модуля ядра')
#     axes[0].plot(epss, np.abs(K))
#     axes[0].grid()
#     axes[1].set_title('График аргумента ядра')
#     axes[1].plot(epss, np.angle(K))
#     axes[1].grid()
#     plt.show()


a = -200
b = 200
n = 1000
p = -2
q = 2
m = 1000
alpha = 0.5
xs = np.linspace(a, b, n)
h_x = xs[1] - xs[0]
epss = np.linspace(p, q, m)
h_eps = epss[1] - epss[0]
plots_f(xs)
plots_F(xs, epss, alpha)



