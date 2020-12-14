"""
Вариант №4
Борисов Д.С.
GL_(2,-3) (r,φ)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import time


def GL_mod(n, p, r, phi):
    return func(r, n, p) * np.exp(1j*p*phi)


def func(r, n, p):
    return np.exp(-r**2) * r**np.abs(p) * L_mod(n, np.abs(p), r**2)


def L_mod(n, p, r):
    result = 0
    for j in range(n+1):
        result += (-1)**j * C(n+p, n-j) * r**j / np.math.factorial(j)
    return result


def C(alpha, k):
    result = 1
    for i in range(k):
        result *= alpha - i
    return result / np.math.factorial(k)


def plots_f(xs, f_xs, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title(f'График модуля {title}')
    axes[0].plot(xs, np.absolute(f_xs))
    axes[0].grid()
    axes[1].set_title(f'График аргумента {title}')
    axes[1].plot(xs, np.angle(f_xs))
    axes[1].grid()
    plt.show()


def plots_image(image):
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15, 5))
    sns.heatmap(np.abs(image), ax=ax1)
    sns.heatmap(np.angle(image), ax=ax2)
    plt.show()


def recovery_image(f, N, m):
    size = 2*N + 1
    matrix = np.zeros((size, size), dtype=np.complex64)

    for j in range(2*N):
        for k in range(2*N):
            alpha = np.round(np.sqrt((j-N)**2 + (k-N)**2)).astype(dtype=np.int32, copy=False)
            if alpha <= N:
                matrix[j, k] = f[alpha] * np.exp(1j * m * np.math.atan2(k - N, j - N))
    return matrix


def manual_integrate(m, xs, f):
    h = xs[1] - xs[0]
    r = np.dot(xs[:, None], xs[None, :])
    A = scipy.special.jv(m, 2 * np.pi * r) * xs[:, None]
    res = (np.dot(f, A) * h).astype(np.complex64)
    res = res * 2 * np.pi / (1j ** m)
    return res


def fft_2d(f, xs, N, M, a):
    h = xs[1] - xs[0]
    padding = int((M - N) / 2)

    y = np.pad(f, padding)
    y = np.fft.fftshift(y)
    Y = np.fft.fft2(y) * (h ** 2)

    Y = np.fft.fftshift(Y)
    result = Y[padding:-padding, padding:-padding]

    b = (N ** 2) / (4 * a * M)

    xv, yv = np.linspace(-b, b, N), np.linspace(-b, b, N)
    new_xs, new_ys = np.meshgrid(xv, yv)
    return result, new_xs, new_ys


def calc_time_fft_and_hankel():
    N = [64, 128, 256, 512]
    n = 2
    m = -3
    R = 5
    a = 5
    b = 5
    for i in N:
        total_time_for_fft = 0
        total_time_for_Hankel = 0
        r = np.linspace(0, R, i, endpoint=True)
        f = func(r, n, m)
        image = recovery_image(f, i - 1, m)
        launches = 5
        for k in range(launches):
            t = time.time()
            manual_integrate(m, r, f)
            t = time.time() - t
            total_time_for_Hankel += t
            new_N = i * 2
            M = int((new_N ** 2) / (4 * b * a))
            t = time.time()
            fft_2d(image, r, new_N, M, R)
            t = time.time() - t
            total_time_for_fft += t
        print("-----------------------------------------------------------------------------------")
        print(f"Среднее время выполнения при дискретизации = {i} точки и при {launches} запусках")
        print(f"Преобразование Ханкеля: {total_time_for_Hankel / launches}")
        print(f"Двумерное БПФ: {total_time_for_fft / launches}")
        print("-----------------------------------------------------------------------------------")


n = 2
m = -3
R = 5
N = 256

r = np.linspace(0, R, N, endpoint=True)
f = func(r, n, m)
plots_f(r, f, "f(r)")
image = recovery_image(f, N - 1, m)
plots_image(image)
man_hank = manual_integrate(m, r, f)
plots_f(r, man_hank, "F(p)")
image_hank = recovery_image(man_hank, N - 1, m)
plots_image(image_hank)

new_N = N * 2
a = 5
b = 5
M = int((new_N ** 2) / (4 * b * a))
fft_f, _, _ = fft_2d(image, r, new_N, M, R)
plots_image(fft_f)

calc_time_fft_and_hankel()
