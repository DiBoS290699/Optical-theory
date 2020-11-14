import numpy as np
import matplotlib.pyplot as plt


def gaussian_beam(x):
    return np.exp(-np.power(x, 2))


def gaussian_beam_2d(x, y):
    return np.exp(-x**2 - y**2)


def rect(t):
    if abs(t) > 0.5:
        return 0
    elif abs(t) < 0.5:
        return 1
    else:
        return 0.5


def rect_2d(t1, t2):
    return rect(t1) * rect(t2)


def fast_fourier_transform(y, b, a, N, M):
    h = (b - a) / (N - 1)
    # Добавление нулей
    zeros = np.zeros(int((M - N) / 2))
    y = np.concatenate((zeros, y, zeros), axis=None)
    # Смена частей вектора
    middle = int(len(y) / 2)
    y = np.concatenate((y[middle:], y[:middle]))
    # БПФ
    Y = np.fft.fft(y, axis=-1) * h
    # Смена частей вектора
    middle = int(len(Y) / 2)
    Y = np.concatenate((Y[middle:], Y[:middle]))
    # Выделение центральных N элементов
    Y = Y[int((M - N) / 2): int((M - N) / 2 + N)]
    # Пересчет области задания функции
    interval = abs(N ** 2 / (4 * a * M))
    return Y, interval


def plots_f(xs, f_xs, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title(f'График модуля {title}')
    axes[0].plot(xs, np.absolute(f_xs))
    axes[0].grid()
    axes[1].set_title(f'График аргумента {title}')
    axes[1].plot(xs, np.angle(f_xs))
    axes[1].grid()
    plt.show()


def compare_plots(xs1, f_xs1, label1, xs2, f_xs2, label2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title('График модуля')
    axes[0].plot(xs1, np.absolute(f_xs1), label=label1)
    axes[0].plot(xs2, np.absolute(f_xs2), label=label2)
    axes[0].grid()
    axes[0].legend()
    axes[1].set_title('График аргумента')
    axes[1].plot(xs1, np.angle(f_xs1), label=label1)
    axes[1].plot(xs2, np.angle(f_xs2), label=label2)
    axes[1].grid()
    axes[1].legend()
    plt.show()


def fast_fourier_transform_2d(Z, b, a, N, M):
    for i in range(N):
        Z[:, i], interval = fast_fourier_transform(Z[:, i], b, a, N, M)
    for i in range(N):
        Z[i, :], interval = fast_fourier_transform(Z[i, :], b, a, N, M)
    return Z, interval


def kernel_of_fourier(x):
    return np.exp(-2 * np.pi * 1j * x)


def manual_integrate(kernel, xs, f):
    h = xs[1] - xs[0]
    A = kernel(np.dot(xs.reshape(-1, 1), xs.reshape(-1, 1).T))
    return np.dot(f, A) * h


def rect_analitic_fourier(x):
    return 1 if x == 0 else np.sin(4 * np.pi * x) / (np.pi * x)


def rect_analitic_fourier_2d(us, vs):
    Z = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Z[i, j] = rect_analitic_fourier(us[i, j]) * rect_analitic_fourier(vs[i, j])
    return Z


N = 512
a = 5
M = 4096

# ------------------------------------ ОДНОМЕРНЫЙ СЛУЧАЙ ------------------------------------
# -------- exp(-x**2) --------
xs = np.linspace(-a, a, N, endpoint=False)
gb_f = gaussian_beam(xs)
fft_gb_f, interval = fast_fourier_transform(gb_f, a, -a, N, M)
xs_interval = np.linspace(-interval, interval, N, endpoint=False)
plots_f(xs, gb_f, "exp(-x**2)")
plots_f(xs_interval, fft_gb_f, "БПФ для exp(-x**2)")

# -------- Сравнение БПФ и стандартного метода --------
man_gb_f = manual_integrate(kernel_of_fourier, xs, gb_f)
compare_plots(xs_interval, fft_gb_f, 'БПФ', xs, man_gb_f, 'Станд. метод')

# -------- rect(x/4) --------
rect_f = []
for x in xs:
    rect_f.append(rect(x/4))
fft_rect_f, interval = fast_fourier_transform(rect_f, a, -a, N, M)
xs_interval = np.linspace(-interval, interval, N, endpoint=False)
plots_f(xs, rect_f, "rect(x/4)")
plots_f(xs_interval, fft_rect_f, "БПФ для rect(x/4)")

# -------- Сравнение БПФ и аналитического результата --------
analit_rect_f = []
for x in xs_interval:
    analit_rect_f.append(rect_analitic_fourier(x))
compare_plots(xs_interval, fft_rect_f, 'БПФ', xs_interval, analit_rect_f, 'Аналитическое решение')

# ------------------------------------ ДВУМЕРНЫЙ СЛУЧАЙ ------------------------------------
# -------- exp(-x**2 - y**2) --------
X, Y = np.meshgrid(xs, xs)
Z = gaussian_beam_2d(X, Y)
fig, arr = plt.subplots(1, 2, figsize=(12, 6))
amp = arr[0].imshow(np.absolute(Z), cmap='hot', interpolation='nearest')
arr[0].set_title('График модуля exp(-x^2-y^2)')
phase = arr[1].imshow(np.angle(Z), cmap='hot', interpolation='nearest')
arr[1].set_title('График аргумента exp(-x^2-y^2)')
fig.colorbar(phase, ax=arr[1])
plt.show()

# -------- БПФ exp(-x**2 - y**2) --------
Z = gaussian_beam_2d(X, Y).astype(np.complex128)
Z_fin_fft, interval = fast_fourier_transform_2d(Z, -a, a, N, M)
xs_interval = np.linspace(-interval, interval, N, endpoint=False)
X, Y = np.meshgrid(xs_interval, xs_interval)
fig, arr = plt.subplots(1, 2, figsize=(12, 6))
amp = arr[0].imshow(np.absolute(Z_fin_fft), cmap='hot', interpolation='nearest')
arr[0].set_title('График модуля БПФ exp(-x^2-y^2)')
fig.colorbar(amp, ax=arr[0])
phase = arr[1].imshow(np.angle(Z_fin_fft), cmap='hot', interpolation='nearest')
arr[1].set_title('График аргумента БПФ exp(-x^2-y^2)')
fig.colorbar(phase, ax=arr[1])
plt.show()

# -------- rect(x/4)*rect(y/4) --------
X, Y = np.meshgrid(xs, xs)
Z = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        Z[i, j] = rect_2d(X[i, j] / 4, Y[i, j] / 4)
fig, arr = plt.subplots(1, 2, figsize=(12, 6))
amp = arr[0].imshow(np.absolute(Z), cmap='hot', interpolation='nearest')
arr[0].set_title('График модуля rect(x/4)*rect(y/4)')
fig.colorbar(amp, ax=arr[0])
phase = arr[1].imshow(np.angle(Z), cmap='hot', interpolation='nearest')
arr[1].set_title('График аргумента rect(x/4)*rect(y/4)')
fig.colorbar(phase, ax=arr[1])
plt.show()

# -------- БПФ rect(x/4)*rect(y/4) --------
X, Y = np.meshgrid(xs, xs)
Z = Z.astype(np.complex128)
Z_fin_fft, interval = fast_fourier_transform_2d(Z, -a, a, N, M)
xs_interval = np.linspace(-interval, interval, N, endpoint=False)
X, Y = np.meshgrid(xs_interval, xs_interval)
fig, arr = plt.subplots(1, 2, figsize=(12, 6))
amp = arr[0].imshow(np.absolute(Z_fin_fft), cmap='hot', interpolation='nearest')
arr[0].set_title('График модуля БПФ rect(x/4)*rect(y/4)')
fig.colorbar(amp, ax=arr[0])
phase = arr[1].imshow(np.angle(Z_fin_fft), cmap='hot', interpolation='nearest')
arr[1].set_title('График аргумента БПФ rect(x/4)*rect(y/4)')
fig.colorbar(phase, ax=arr[1])
plt.show()

# -------- Аналитическое решение --------
Z = rect_analitic_fourier_2d(X, Y)
fig, arr = plt.subplots(1, 2, figsize=(12, 6))
arr[0].imshow(np.absolute(Z), cmap='hot', interpolation='nearest')
arr[0].set_title('График модуля аналитики')
fig.colorbar(amp, ax=arr[0])
phase = arr[1].imshow(np.angle(Z), cmap='hot', interpolation='nearest')
arr[1].set_title('График аргумента аналитики')
fig.colorbar(phase, ax=arr[1])
plt.show()
