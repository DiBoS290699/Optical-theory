import numpy as np
import matplotlib.pyplot as plt


def gaussian_beam(x):
    return np.exp(-np.power(x, 2))


def gaussian_beam_2d(x, y):
    return np.exp(-x**2 - y**2)


def rect(t_mas):
    if abs(t_mas) > 0.5:
        return 0
    elif abs(t_mas) < 0.5:
        return 1
    else:
        return 0.5


def rect_2d(t1, t2):
    return rect(t1) * rect(t2)


def discret_func(f_x, N, a, b):
    xs = np.linspace(a, b, N, endpoint=False)
    y = []
    for x in xs:
        y.append(f_x(x))
    return xs, np.array(y)


def discret_func_2d(f_x_y, N, a, b):
    xs = np.linspace(a, b, N, endpoint=False)
    f = []
    for i in range(N):
        line = []
        for j in range(N):
            line.append(f_x_y(xs[i], xs[j]))
        f.append(line)
    return xs, np.array(f, dtype=np.complex128)


def fast_fourier_transform(f, N, M, a):
    h_x = 2*a / (N - 1)

    zeros = np.zeros(int((M - N) / 2))
    f = np.concatenate((zeros, f, zeros))
    middle = int(len(f) / 2)
    f = np.concatenate((f[middle:], f[:middle]), axis=None)

    f = np.fft.fft(f, axis=-1) * h_x

    middle = int(len(f) / 2)
    f = np.concatenate((f[middle:], f[:middle]))

    number_zeros = (M - N) // 2
    if number_zeros != 0:
        fft_result = f[number_zeros:-number_zeros]
    else:
        fft_result = f

    b = (N ** 2) / (4 * a * M)
    new_xs = np.linspace(-b, b, N, endpoint=False)
    return new_xs, np.array(fft_result)


def fast_fourier_transform_2d(f, N, M, a):
    new_xs = []
    for i in range(len(f)):
        new_xs, f[i, :] = fast_fourier_transform(f[i, :], N, M, a)
    for j in range(len(f[0])):
        new_xs, f[:, j] = fast_fourier_transform(f[:, j], N, M, a)
    return new_xs, f


def plots_f(xs, f_xs, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title(f'График модуля {title}')
    axes[0].plot(xs, np.abs(f_xs))
    axes[0].grid()
    axes[1].set_title(f'График аргумента {title}')
    axes[1].plot(xs, np.angle(f_xs))
    axes[1].grid()
    plt.show()


def plots_f_2d(f_xs, title):
    fig, arr = plt.subplots(1, 2, figsize=(12, 6))
    amp = arr[0].imshow(np.absolute(f_xs), cmap='hot', interpolation='nearest')
    arr[0].set_title(f'График модуля {title}')
    phase = arr[1].imshow(np.angle(f_xs), cmap='hot', interpolation='nearest')
    arr[1].set_title(f'График аргумента {title}')
    fig.colorbar(phase, ax=arr[1])
    plt.show()


def compare_plots(xs1, f_xs1, label1, xs2, f_xs2, label2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title('График модуля')
    axes[0].plot(xs1, np.abs(f_xs1), label=label1)
    axes[0].plot(xs2, np.abs(f_xs2), label=label2)
    axes[0].grid()
    axes[0].legend()
    axes[1].set_title('График аргумента')
    axes[1].plot(xs1, np.angle(f_xs1), label=label1)
    axes[1].plot(xs2, np.angle(f_xs2), label=label2)
    axes[1].grid()
    axes[1].legend()
    plt.show()


def kernel_of_fourier(x):
    return np.exp(-2 * np.pi * 1j * x)


def manual_integrate(kernel, xs, f):
    h = xs[1] - xs[0]
    A = kernel(np.dot(xs.reshape(-1, 1), xs.reshape(-1, 1).T))
    return np.dot(f, A) * h


def rect_analitic_fourier(xs):
    y = []
    for x in xs:
        y.append(1 if x == 0 else np.sin(4*np.pi*x) / (np.pi * x))
    return y


N = 516
a = 5
M = 4096

# ------------------------------------ ОДНОМЕРНЫЙ СЛУЧАЙ ------------------------------------
# -------- exp(-x**2) --------
gb_xs, gb_f = discret_func(gaussian_beam, N, -a, a)
fft_gb_xs, fft_gb_f = fast_fourier_transform(gb_f, N, M, a)
plots_f(gb_xs, gb_f, "exp(-x**2)")
plots_f(fft_gb_xs, fft_gb_f, "БПФ для exp(-x**2)")

# -------- Сравнение БПФ и стандартного метода --------
man_gb_f = manual_integrate(kernel_of_fourier, gb_xs, gb_f)
compare_plots(fft_gb_xs, fft_gb_f, 'БПФ', gb_xs, man_gb_f, 'Станд. метод')

# -------- rect(x/4) --------
rect_xs, rect_f = discret_func(rect, N, -a, a)
fft_rect_xs, fft_rect_f = fast_fourier_transform(rect_f, N, M, a)
plots_f(rect_xs, rect_f, "rect(x/4)")
plots_f(fft_rect_xs, fft_rect_f, "БПФ для rect(x/4)")

# -------- Сравнение БПФ и аналитического результата --------
analit_rect_f = rect_analitic_fourier(fft_rect_xs)
compare_plots(fft_rect_xs, fft_rect_f, 'БПФ', fft_rect_xs, analit_rect_f, 'Аналитическое решение')

# ------------------------------------ ДВУМЕРНЫЙ СЛУЧАЙ ------------------------------------
# -------- exp(-x**2 - y**2) --------
gb_xs, gb_f = discret_func_2d(gaussian_beam_2d, N, -a, a)
fft_gb_xs, fft_gb_f = fast_fourier_transform_2d(gb_f, N, M, a)
plots_f_2d(gb_f, "exp(-x**2 - y**2)")
plots_f_2d(fft_gb_f, "БПФ для exp(-x**2 - y**2)")

# -------- rect(x/4)*rect(y/4) --------
rect_xs, rect_f = discret_func_2d(rect_2d, N, -a, a)
fft_rect_xs, fft_rect_f = fast_fourier_transform_2d(rect_f, N, M, a)
plots_f_2d(rect_f, "rect(x/4)*rect(y/4)")
plots_f_2d(fft_rect_f, "БПФ для rect(x/4)*rect(y/4)")

