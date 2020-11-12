import numpy as np
import matplotlib.pyplot as plt


def gaussian_beam(x):
    return np.exp(-np.power(x, 2))


def rect(t_mas):
    y = []
    for t in t_mas:
        if abs(t) > 0.5:
            y.append(0)
        elif abs(t) < 0.5:
            y.append(1)
        else:
            y.append(0.5)
    return y


def discret_func(f_x, N, a, b):
    x = np.linspace(a, b, N)
    y = f_x(x)
    return np.array(x), np.array(y)


def add_zeros(f, M):
    number_zeros = (M - len(f)) // 2
    new_f = np.zeros(M)
    new_f[number_zeros:number_zeros + len(f)] = f
    return np.array(new_f)


def swap_half_array(array):
    new_array = []
    index_half_array = len(array) // 2
    new_array.extend(array[index_half_array:])
    new_array.extend(array[:index_half_array])
    return np.array(new_array)


def fast_fourier_transform(f, xs, N, M, a):
    h_x = xs[1] - xs[0]

    new_f = add_zeros(f, M)
    new_f = swap_half_array(new_f)
    res_f = np.fft.fft(new_f) * h_x

    res_f = swap_half_array(res_f)
    number_zeros = (M - N) // 2
    if number_zeros != 0:
        fft_result = res_f[number_zeros:-number_zeros]
    else:
        fft_result = res_f
    # centre_res_f = res_f[number_zeros:-number_zeros]

    b = (N ** 2) / (4 * a * M)
    new_xs = np.linspace(-b, b, N)
    return new_xs, fft_result


def plots_f(xs, f_xs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title('График модуля исходного сигнала')
    axes[0].plot(xs, np.abs(f_xs))
    axes[0].grid()
    axes[1].set_title('График аргумента исходного сигнала')
    axes[1].plot(xs, np.angle(f_xs))
    axes[1].grid()
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


def manual_integrate(kernel, f, xs):
    h = xs[1] - xs[0]
    A = kernel(np.dot(xs.reshape(-1, 1), xs.reshape(-1, 1).T))
    return np.dot(f, A) * h


def rect_analitic_fourier(xs):
    y = []
    for x in xs:
        y.append(1 if x == 0 else np.sin(4*np.pi*x) / (np.pi * x))
    return y


N = 100
a = 5
M = int((N ** 2) / (4 * a**2))
#M = 16
rect_xs, rect_f = discret_func(rect, N, -a, a)
# gb_xs, gb = discret_func(gaussian_beam, N, -a, a)
# new_gb = add_zeros(gb, M)
# fourier_xs, fourier_f = fast_fourier_transform(gb, gb_xs, N, M, a)
# manual_f = manual_integrate(kernel_of_fourier, gb, gb_xs)
# plots_f(gb_xs, gb)
# plots_f(fourier_xs, fourier_f)
# compare_plots(gb_xs, manual_f, fourier_xs, fourier_f)

rect_fourier_xs, rect_fourier_f = fast_fourier_transform(rect_f, rect_xs, N, M, a)
plots_f(rect_fourier_xs, rect_fourier_f)
rect_analitic_f = rect_analitic_fourier(rect_xs)
compare_plots(rect_xs, rect_analitic_f, "Аналитическое", rect_fourier_xs, rect_fourier_f, "numpy.fft")
# manual_f = manual_integrate(kernel_of_fourier, rect_f, rect_xs)
# compare_plots(rect_xs, manual_f, "Численное", rect_fourier_xs, rect_fourier_f, "numpy.fft")




