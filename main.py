#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ЛАБОРАТОРНАЯ РАБОТА № 2
Дисциплина: «Вычислительная математика»
Вариант 10.

Реализованы четыре задания:

  Задание 1 (п. 6.3)  - интерполяция таблично заданной функции
                         полиномом Ньютона в разделённых разностях;
  Задание 2 (п. 6.4)  - численное дифференцирование: первая и вторая
                         производные полинома Ньютона;
  Задание 3.1 (п. 6.5) - квадратурная формула трапеций с автоматическим
                         выбором шага интегрирования;
  Задание 3.2 (п. 6.5) - вычисление интеграла формулой Гаусса.

Данные варианта 10:

  Задания 1 и 2 (пп. 6.3, 6.4):
      y = sin^2(x) + 1,   x in [0; pi/2],   h = pi/20;
      исходная сетка  x_i = i*h,       i = 0..10  (11 узлов);
      новая сетка     x_j = j*h/2,     j = 0..20  (21 узел).

  Задание 3 (п. 6.5, вариант 10):
      I = integral from 0 to 2 of exp(-x^2)*cos(x) dx,   n = 6.

Программа не использует сторонние библиотеки (только стандартный модуль
math), весь численный аппарат написан с нуля.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple


# ===========================================================================
#  ДАННЫЕ ВАРИАНТА 10
# ===========================================================================

# --- задания 1 и 2: таблично заданная функция -----------------------------
X_START = 0.0
X_END = math.pi / 2.0
STEP_H = math.pi / 20.0

# исходная сетка узлов интерполяции: x_i = i*h, i = 0..10
BASE_NODES: List[float] = [i * STEP_H for i in range(11)]
# новая сетка: x_j = j*h/2, j = 0..20
NEW_NODES: List[float] = [j * STEP_H / 2.0 for j in range(21)]


def exact_function(x: float) -> float:
    """Интерполируемая функция y = sin^2(x) + 1."""
    return math.sin(x) ** 2 + 1.0


def exact_first(x: float) -> float:
    """Точная первая производная y' = sin(2x)."""
    return math.sin(2.0 * x)


def exact_second(x: float) -> float:
    """Точная вторая производная y'' = 2*cos(2x)."""
    return 2.0 * math.cos(2.0 * x)


def exact_derivative(k: int, x: float) -> float:
    """
    Точная k-я производная функции (k >= 1).

    Представление:  y(x) = 3/2 - (1/2)*cos(2x),  откуда
        y^(k)(x) = -(1/2)*2^k*cos(2x + k*pi/2),  k >= 1.
    """
    if k <= 0:
        return exact_function(x)
    return -0.5 * (2.0 ** k) * math.cos(2.0 * x + k * math.pi / 2.0)


def derivative_bound(k: int) -> float:
    """
    Глобальная оценка сверху |y^(k)(x)| на всей вещественной прямой.

        k = 0 :  |y|     <= 3/2;
        k >= 1:  |y^(k)| <= 2^(k-1).
    """
    if k <= 0:
        return 1.5
    return 2.0 ** (k - 1)


# --- задание 3: подынтегральная функция -----------------------------------
INTEGRAL_A = 0.0
INTEGRAL_B = 2.0
INTEGRAL_N_ORDER = 6          # порядок формулы Гаусса по варианту


def integrand(x: float) -> float:
    """Подынтегральная функция f(x) = exp(-x^2)*cos(x)."""
    return math.exp(-x * x) * math.cos(x)


def integrand_second(x: float) -> float:
    """
    Вторая производная подынтегральной функции:
        f'(x)  = exp(-x^2)*(-2x*cos(x) - sin(x));
        f''(x) = exp(-x^2)*((4x^2 - 3)*cos(x) + 4x*sin(x)).
    """
    exp_part = math.exp(-x * x)
    return exp_part * ((4.0 * x * x - 3.0) * math.cos(x) + 4.0 * x * math.sin(x))


# ===========================================================================
#  ОБЩИЕ ПРОЦЕДУРЫ ВВОДА-ВЫВОДА
# ===========================================================================

def _input(prompt: str) -> str:
    """Обёртка над input(): корректно завершает работу при закрытии потока ввода."""
    try:
        return input(prompt)
    except EOFError:
        raise SystemExit("\nВвод прерван. Работа программы завершена.")


def read_float(prompt: str, default: float) -> float:
    """Читает вещественное число; пустая строка означает значение по умолчанию."""
    while True:
        raw = _input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("  Ошибка: введите вещественное число (например, 1e-6).")


def read_int(prompt: str, default: int, lo: int, hi: int) -> int:
    """Читает целое число из диапазона [lo; hi]; пустая строка - значение по умолчанию."""
    while True:
        raw = _input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            value = -1
        if lo <= value <= hi:
            return value
        print(f"  Ошибка: требуется целое число от {lo} до {hi}.")


def read_yes_no(prompt: str, default: bool = True) -> bool:
    """Спрашивает подтверждение вида «да/нет»."""
    hint = "Y/n" if default else "y/N"
    while True:
        raw = _input(f"{prompt} [{hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes", "д", "да", "+"):
            return True
        if raw in ("n", "no", "н", "нет", "-"):
            return False
        print("  Ошибка: ответьте «да» или «нет».")


def print_table(header: Sequence[str], rows: Sequence[Sequence[str]],
                indent: str = "  ") -> None:
    """Печатает таблицу с выравниванием колонок по ширине."""
    widths = [len(str(h)) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def line(cells: Sequence[str]) -> str:
        return indent + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    print(line(header))
    print(indent + "  ".join("-" * w for w in widths))
    for row in rows:
        print(line(row))


def print_matrix(title: str, matrix: Sequence[Sequence[float]],
                 precision: int = 8) -> None:
    """Печатает матрицу с фиксированной точностью."""
    print(title)
    for row in matrix:
        print("    " + "  ".join(f"{v: .{precision}f}" for v in row))


def section(title: str) -> None:
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)


# ===========================================================================
#  ЗАДАНИЕ 1. ИНТЕРПОЛЯЦИЯ ПОЛИНОМОМ НЬЮТОНА
# ===========================================================================

def divided_differences(xs: Sequence[float], fs: Sequence[float]) -> List[float]:
    """
    Разделённые разности f[x_0..x_k] для k = 0..m.

    Схема: на k-м шаге вычисляется только новая диагональ
        c_k = (c_{k-1} - c_{k-1}^{пред}) / (x_k - x_{k-(k-1)}),
    поэтому массив коэффициентов перезаписывается «на месте».
    """
    coeffs = list(fs)
    size = len(xs)
    for k in range(1, size):
        for i in range(size - 1, k - 1, -1):
            coeffs[i] = (coeffs[i] - coeffs[i - 1]) / (xs[i] - xs[i - k])
    return coeffs


def select_window(nodes: Sequence[float], order: int, x: float) -> List[float]:
    """
    Выбирает (order+1) узлов исходной сетки для локального полинома.

    Предпочтение отдаётся окну, содержащему точку x; при равенстве
    выбирается окно с наименьшим расстоянием от x до его центра.
    """
    width = order + 1
    if width > len(nodes):
        raise ValueError(f"Порядок полинома {order} требует {width} узлов, "
                         f"а в исходной сетке только {len(nodes)}.")
    best_key: Optional[Tuple[int, float]] = None
    best_window: List[float] = []
    for start in range(0, len(nodes) - width + 1):
        window = list(nodes[start:start + width])
        inside = window[0] <= x <= window[-1]
        centre = 0.5 * (window[0] + window[-1])
        key = (0 if inside else 1, abs(x - centre))
        if best_key is None or key < best_key:
            best_key = key
            best_window = window
    return best_window


def newton_basis(xs: Sequence[float], x: float) -> Tuple[List[float], List[float], List[float]]:
    """
    Значения базисных многочленов Ньютона и их первых двух производных в точке x.

        w_0 = 1,   w_k(x) = product_{j=0..k-1} (x - x_j);
        w'_k  = w'_{k-1}*(x - x_{k-1}) + w_{k-1};
        w''_k = w''_{k-1}*(x - x_{k-1}) + 2*w'_{k-1}.
    """
    size = len(xs)
    w = [1.0] * size
    wp = [0.0] * size
    wpp = [0.0] * size
    for k in range(1, size):
        d = x - xs[k - 1]
        wpp[k] = wpp[k - 1] * d + 2.0 * wp[k - 1]
        wp[k] = wp[k - 1] * d + w[k - 1]
        w[k] = w[k - 1] * d
    return w, wp, wpp


def evaluate_newton(coeffs: Sequence[float], xs: Sequence[float], x: float) -> Tuple[float, float, float]:
    """
    Значения полинома Ньютона, его первой и второй производной в точке x:

        P(x)   = sum c_k * w_k(x);
        P'(x)  = sum c_k * w'_k(x);
        P''(x) = sum c_k * w''_k(x).
    """
    w, wp, wpp = newton_basis(xs, x)
    value = 0.0
    first = 0.0
    second = 0.0
    for k, c_k in enumerate(coeffs):
        value += c_k * w[k]
        first += c_k * wp[k]
        second += c_k * wpp[k]
    return value, first, second


def omega_and_derivatives(xs: Sequence[float], x: float) -> Tuple[float, float, float]:
    """
    Узловой многочлен omega(x) = product (x - x_i) и его две производные.

    Многочлен сначала приводится к стандартному виду (умножение на
    линейные множители), затем значения вычисляются схемой Горнера
    одновременно для значения и двух производных.
    """
    poly = [1.0]                       # коэффициенты при возрастании степени
    for xi in xs:
        grown = [0.0] * (len(poly) + 1)
        for p, c in enumerate(poly):
            grown[p + 1] += c
            grown[p] -= c * xi
        poly = grown

    degree = len(poly) - 1
    value = poly[degree]
    first = 0.0
    second = 0.0
    for p in range(degree - 1, -1, -1):
        second = second * x + 2.0 * first
        first = first * x + value
        value = value * x + poly[p]
    return value, first, second


def factorial(number: int) -> int:
    """Факториал без использования math.factorial (для наглядности счёта)."""
    result = 1
    for i in range(2, number + 1):
        result *= i
    return result


@dataclass
class ApproximationResult:
    """Результат приближения функции или её производных в одной точке."""
    x: float
    nodes: List[float]
    value: float
    first: float
    second: float
    exact: float
    exact_first: float
    exact_second: float
    err_value: float
    err_first: float
    err_second: float
    bound_value: float
    bound_first: float
    bound_second: float


def approximate_at(order: int, x: float) -> ApproximationResult:
    """
    Строит локальный полином Ньютона степени `order` по (order+1) ближайшим
    узлам исходной сетки и возвращает приближённые значения функции и её
    двух производных вместе с оценками погрешности.
    """
    nodes = select_window(BASE_NODES, order, x)
    values = [exact_function(t) for t in nodes]
    coeffs = divided_differences(nodes, values)

    value, first, second = evaluate_newton(coeffs, nodes, x)
    omega, omega1, omega2 = omega_and_derivatives(nodes, x)

    m = order
    denom = factorial(m + 1)
    m1 = derivative_bound(m + 1)
    m2 = derivative_bound(m + 2)
    m3 = derivative_bound(m + 3)

    bound_value = m1 * abs(omega) / denom
    bound_first = (m2 * abs(omega) + m1 * abs(omega1)) / denom
    bound_second = (m3 * abs(omega) + 2.0 * m2 * abs(omega1)
                    + m1 * abs(omega2)) / denom

    return ApproximationResult(
        x=x,
        nodes=nodes,
        value=value,
        first=first,
        second=second,
        exact=exact_function(x),
        exact_first=exact_first(x),
        exact_second=exact_second(x),
        err_value=abs(exact_function(x) - value),
        err_first=abs(exact_first(x) - first),
        err_second=abs(exact_second(x) - second),
        bound_value=bound_value,
        bound_first=bound_first,
        bound_second=bound_second,
    )


def run_task_1() -> None:
    """Задание 1 (п. 6.3): интерполяция таблично заданной функции."""
    section("ЗАДАНИЕ 1. ИНТЕРПОЛЯЦИЯ ТАБЛИЧНО ЗАДАННОЙ ФУНКЦИИ ПОЛИНОМОМ НЬЮТОНА")
    print("  Функция варианта 10: y = sin^2(x) + 1,  x in [0; pi/2],  h = pi/20.")
    print(f"  Исходная сетка ({len(BASE_NODES)} узлов): x_i = i*h, i = 0..10.")
    print(f"  Новая сетка    ({len(NEW_NODES)} узлов): x_j = j*h/2, j = 0..20.")
    print("")
    print("  Входные данные: исходная сетка, значения функции, новая сетка,")
    print("                  порядок полинома интерполяции.")

    order = read_int("  Порядок полинома m (1..10)", 3, 1, len(BASE_NODES) - 1)

    started = time.perf_counter()
    print("")
    print(f"  Этап 1. Значения функции на исходной сетке (степень полинома m = {order})")
    rows = []
    for i, node in enumerate(BASE_NODES):
        rows.append((str(i), f"{node:.10f}", f"{exact_function(node):.10f}"))
    print_table(("i", "x_i", "y_i = f(x_i)"), rows)

    print("")
    print(f"  Этап 2. Значения полинома степени m = {order} на новой сетке")
    results: List[ApproximationResult] = []
    for x_new in NEW_NODES:
        results.append(approximate_at(order, x_new))

    rows = []
    for j, res in enumerate(results):
        rows.append((
            str(j),
            f"{res.x:.10f}",
            f"{res.exact:.10f}",
            f"{res.value:.10f}",
            f"{res.err_value:.3e}",
            f"{res.bound_value:.3e}",
        ))
    print_table(("j", "x_j", "y(x_j)", "P_m(x_j)", "|y - P_m|", "Оценка"), rows)

    elapsed = (time.perf_counter() - started) * 1e3

    worst = max(results, key=lambda r: r.err_value)
    worst_index = results.index(worst)
    print("")
    print("  Пример построения полинома (точка новой сетки с индексом "
          f"{worst_index}):")
    print(f"    x = {worst.x:.10f}")
    print(f"    Выбранные узлы: " + ", ".join(f"{t:.6f}" for t in worst.nodes))
    nodes_fs = [exact_function(t) for t in worst.nodes]
    coeffs = divided_differences(worst.nodes, nodes_fs)
    print("  Таблица разделённых разностей:")
    rows = []
    for k, (t, c_k) in enumerate(zip(worst.nodes, coeffs)):
        rows.append((str(k), f"{t:.10f}", f"{c_k:.10f}"))
    print_table(("k", "x_k", "f[x_0..x_k]"), rows)
    terms = "  +  ".join(f"({c_k:.6e})*prod_{{j<{k}}}(x - x_j)" for k, c_k in enumerate(coeffs))
    print(f"    P_m(x) = {terms}")

    print("")
    print("  Итоги задания 1:")
    print(f"    Максимальная фактическая погрешность |y - P_m|  = "
          f"{max(r.err_value for r in results):.6e}")
    print(f"    Максимальная оценка погрешности                = "
          f"{max(r.bound_value for r in results):.6e}")
    print("    Проверка: фактическая погрешность не превышает оценку "
          f"{'выполнена' if all(r.err_value <= r.bound_value + 1e-15 for r in results) else 'НЕ ВЫПОЛНЕНА'}")
    print(f"    Время выполнения: {elapsed:.3f} мс")


def run_task_2() -> None:
    """Задание 2 (п. 6.4): численное дифференцирование."""
    section("ЗАДАНИЕ 2. ЧИСЛЕННОЕ ДИФФЕРЕНЦИРОВАНИЕ ПОЛИНОМОМ НЬЮТОНА")
    print("  Функция варианта 10: y = sin^2(x) + 1,  x in [0; pi/2],  h = pi/20.")
    print("  Точные производные:  y' = sin(2x),  y'' = 2*cos(2x).")
    print("")
    print("  Входные данные: исходная сетка, значения функции, новая сетка,")
    print("                  порядок полинома.")

    order = read_int("  Порядок полинома m (1..10)", 3, 1, len(BASE_NODES) - 1)

    started = time.perf_counter()
    results: List[ApproximationResult] = []
    for x_new in NEW_NODES:
        results.append(approximate_at(order, x_new))

    print("")
    print(f"  Этап 1. Первая производная: сравнение P'_m(x) с точным y'(x)")
    rows = []
    for j, res in enumerate(results):
        rows.append((
            str(j), f"{res.x:.8f}", f"{res.exact_first:.8f}", f"{res.first:.8f}",
            f"{res.err_first:.3e}", f"{res.bound_first:.3e}",
        ))
    print_table(("j", "x_j", "y'(x_j)", "P'_m(x_j)", "|y' - P'_m|", "Оценка"), rows)

    print("")
    print(f"  Этап 2. Вторая производная: сравнение P''_m(x) с точным y''(x)")
    rows = []
    for j, res in enumerate(results):
        rows.append((
            str(j), f"{res.x:.8f}", f"{res.exact_second:.8f}", f"{res.second:.8f}",
            f"{res.err_second:.3e}", f"{res.bound_second:.3e}",
        ))
    print_table(("j", "x_j", "y''(x_j)", "P''_m(x_j)", "|y'' - P''_m|", "Оценка"), rows)

    elapsed = (time.perf_counter() - started) * 1e3

    print("")
    print("  Формулы производных полинома Ньютона, использованные в программе:")
    print("    P(x)   = sum_{k=0..m} c_k * w_k(x),      w_k = prod_{j<k}(x - x_j);")
    print("    P'(x)  = sum c_k * [w'_{k-1}*(x - x_{k-1}) + w_{k-1}];")
    print("    P''(x) = sum c_k * [w''_{k-1}*(x - x_{k-1}) + 2*w'_{k-1}].")
    print("")
    print("  Оценки погрешности дифференцирования (M_k = 2^(k-1), k >= 1):")
    print("    |y'  - P'_m|  <= [M_{m+2}*|w| + M_{m+1}*|w'|] / (m+1)!;")
    print("    |y'' - P''_m| <= [M_{m+3}*|w| + 2*M_{m+2}*|w'| + M_{m+1}*|w''|] / (m+1)!.")
    print("")
    print("  Итоги задания 2:")
    print(f"    Максимальная погрешность первой производной  = "
          f"{max(r.err_first for r in results):.6e}")
    print(f"    Максимальная погрешность второй производной   = "
          f"{max(r.err_second for r in results):.6e}")
    print(f"    Максимальная оценка для первой производной    = "
          f"{max(r.bound_first for r in results):.6e}")
    print(f"    Максимальная оценка для второй производной    = "
          f"{max(r.bound_second for r in results):.6e}")
    ok_first = all(r.err_first <= r.bound_first + 1e-12 for r in results)
    ok_second = all(r.err_second <= r.bound_second + 1e-12 for r in results)
    print(f"    Проверка оценок: первая производная - "
          f"{'выполнена' if ok_first else 'НЕ ВЫПОЛНЕНА'}, вторая - "
          f"{'выполнена' if ok_second else 'НЕ ВЫПОЛНЕНА'}")
    print(f"    Время выполнения: {elapsed:.3f} мс")


# ===========================================================================
#  ЗАДАНИЕ 3.1. ФОРМУЛА ТРАПЕЦИЙ С АВТОМАТИЧЕСКИМ ВЫБОРОМ ШАГА
# ===========================================================================

def trapezoid_rule(f: Callable[[float], float], a: float, b: float, parts: int) -> float:
    """
    Составная формула трапеций:

        I_h = h * [ f(a)/2 + f(x_1) + ... + f(x_{n-1}) + f(b)/2 ],
        h = (b - a)/n.
    """
    if parts < 1:
        raise ValueError("Число интервалов должно быть не меньше 1.")
    h = (b - a) / parts
    total = 0.5 * (f(a) + f(b))
    for k in range(1, parts):
        total += f(a + k * h)
    return total * h


def max_abs_second(f_second: Callable[[float], float], a: float, b: float,
                   samples: int = 20001) -> Tuple[float, float]:
    """Максимум |f''(x)| на отрезке [a; b] по равномерной пробной сетке."""
    best_value = 0.0
    best_x = a
    step = (b - a) / (samples - 1)
    for i in range(samples):
        x = a + i * step
        value = abs(f_second(x))
        if value > best_value:
            best_value = value
            best_x = x
    return best_value, best_x


def run_task_3_1() -> None:
    """Задание 3.1 (п. 6.5): квадратурная формула трапеций с автовыбором шага."""
    section("ЗАДАНИЕ 3.1. ИНТЕГРИРОВАНИЕ ФОРМУЛОЙ ТРАПЕЦИЙ (АВТОВЫБОР ШАГА)")
    print(f"  Интеграл варианта 10: I = integral from {INTEGRAL_A} to {INTEGRAL_B} "
          f"of exp(-x^2)*cos(x) dx")
    print("  Подынтегральная функция: f(x) = exp(-x^2)*cos(x)")
    print("")
    print("  Входные данные: начальное количество интервалов n0, границы")
    print("  интервала, подынтегральная функция, относительная точность.")

    n0 = read_int("  Начальное число интервалов n0", 6, 1, 100000)
    rel_tol = read_float("  Относительная точность eps", 1e-6)
    if rel_tol <= 0.0:
        print("  Точность заменена на значение по умолчанию 1e-6.")
        rel_tol = 1e-6

    started = time.perf_counter()
    a, b = INTEGRAL_A, INTEGRAL_B
    max_n = 1 << 22

    parts = n0
    value_prev = trapezoid_rule(integrand, a, b, parts)
    # первая строка журнала: результат на исходной сетке (оценка ещё нет)
    history: List[Tuple[int, float, float, float]] = [(parts, value_prev, float("nan"), float("nan"))]
    value_curr = value_prev
    parts_curr = parts

    while parts <= max_n // 2:
        parts *= 2
        value_curr = trapezoid_rule(integrand, a, b, parts)
        # правило Ричардсона для трапеций: погрешость ~ C*h^2,
        # поэтому |I_2n - I_n| / 3 есть оценка погрешости I_2n
        estimate = abs(value_curr - value_prev) / 3.0
        relative = estimate / max(abs(value_curr), 1e-300)
        history.append((parts, value_curr, estimate, relative))
        value_prev = value_curr
        parts_curr = parts
        if relative <= rel_tol:
            break

    elapsed = (time.perf_counter() - started) * 1e3

    print("")
    print("  Этап 1. Автоматический выбор шага интегрирования")
    rows = []
    for n_int, val, est, rel in history:
        mark = "*" if rel <= rel_tol else ""
        est_text = "  --    " if math.isnan(est) else f"{est:.3e}"
        rel_text = "  --    " if math.isnan(rel) else f"{rel:.3e}"
        rows.append((str(n_int), f"{val:.12f}", est_text, rel_text, mark))
    print_table(("n", "I_n", "|I_2n - I_n|/3", "отн. оценка", "готово"), rows)
    print("  Первая строка - результат на исходной сетке с n0 интервалов,")
    print("  звёздочкой отмечен ряд, в котором достигнута заданная точность.")

    h_final = (b - a) / parts_curr
    m_second, x_at_max = max_abs_second(integrand_second, a, b)
    theory_bound = (b - a) * h_final ** 2 / 12.0 * m_second

    reference = adaptive_simpson(integrand, a, b, 1e-14)
    achieved = abs(value_curr - reference) / max(abs(reference), 1e-300)

    print("")
    print("  Этап 2. Итоговые результаты")
    print(f"    Значение интеграла I            = {value_curr:.12f}")
    print(f"    Конечное число интервалов n     = {parts_curr}")
    print(f"    Шаг сетки h = (b - a)/n         = {h_final:.6e}")
    print(f"    Оценка точности (Ричардсон)     = "
          f"{history[-1][2]:.3e} (отн. {history[-1][3]:.3e})")
    print(f"    Теоретическая оценка (b-a)h^2/12*max|f''| = {theory_bound:.3e}")
    print(f"    max|f''(x)| = {m_second:.6f} при x = {x_at_max:.6f}")
    print(f"    Контрольное значение (адаптивный Симпсон) = {reference:.12f}")
    print(f"    Достигнутая относительная погрешность     = {achieved:.3e}")
    print(f"    Заданная относительная точность           = {rel_tol:.3e}")
    verdict = "выполнена" if achieved <= rel_tol else "не выполнена"
    print(f"    Требование точности: {verdict}")
    print(f"    Время выполнения: {elapsed:.3f} мс")


# ===========================================================================
#  ЗАДАНИЕ 3.2. ФОРМУЛА ГАУССА
# ===========================================================================

def legendre_polynomial(order: int, t: float) -> float:
    """
    Многочлен Лежандра P_order(t) по трёхчленной рекурренте:
        P_0 = 1, P_1 = t,
        (k+1)*P_{k+1} = (2k+1)*t*P_k - k*P_{k-1}.
    """
    if order == 0:
        return 1.0
    if order == 1:
        return t
    pm1, p = 1.0, t
    for k in range(1, order):
        pm1, p = p, ((2 * k + 1) * t * p - k * pm1) / (k + 1)
    return p


def legendre_derivative(order: int, t: float) -> float:
    """Производная многочлена Лежандра: P'_n = n*(t*P_n - P_{n-1})/(t^2 - 1)."""
    if order == 0:
        return 0.0
    p_n = legendre_polynomial(order, t)
    p_nm1 = legendre_polynomial(order - 1, t)
    if abs(t - 1.0) < 1e-14 or abs(t + 1.0) < 1e-14:
        return order * (order + 1) / 2.0 * (1.0 if t > 0 else (-1.0) ** order)
    return order * (t * p_n - p_nm1) / (t * t - 1.0)


def gauss_nodes_weights(order: int) -> Tuple[List[float], List[float]]:
    """
    Узлы и веса формулы Гаусса - Лежандра степени `order` на отрезке [-1; 1].

    Корни многочлена P_order(t) уточняются методом Ньютона от начального
    приближения t_i = cos(pi*(i + 0.75)/(order + 0.5)); веса вычисляются
    по формуле  w_i = 2 / ((1 - t_i^2) * [P'_order(t_i)]^2).
    """
    if order < 1:
        raise ValueError("Порядок формулы Гаусса должен быть не меньше 1.")
    roots: List[float] = []
    for i in range(order):
        t = math.cos(math.pi * (i + 0.75) / (order + 0.5))
        for _ in range(100):
            p_val = legendre_polynomial(order, t)
            d_val = legendre_derivative(order, t)
            if abs(d_val) < 1e-300:
                break
            step = p_val / d_val
            t -= step
            if abs(step) < 1e-16:
                break
        roots.append(t)

    weights: List[float] = []
    for t in roots:
        d_val = legendre_derivative(order, t)
        weights.append(2.0 / ((1.0 - t * t) * d_val * d_val))
    return roots, weights


def gauss_integral(f: Callable[[float], float], a: float, b: float,
                   order: int, parts: int) -> float:
    """
    Составная формула Гаусса: отрезок [a; b] делится на `parts` равных
    частей, на каждой применяется формула Гаусса порядка `order`.
    """
    nodes, weights = gauss_nodes_weights(order)
    width = (b - a) / parts
    total = 0.0
    for k in range(parts):
        left = a + k * width
        centre = left + 0.5 * width
        half = 0.5 * width
        for t, w in zip(nodes, weights):
            total += w * f(centre + half * t)
    return total * half


def adaptive_simpson(f: Callable[[float], float], a: float, b: float,
                     tol: float, depth: int = 0) -> float:
    """
    Адаптивная формула Симпсона - независимый контрольный счёт
    интеграла с точностью tol.
    """
    centre = 0.5 * (a + b)
    whole = (b - a) / 6.0 * (f(a) + 4.0 * f(centre) + f(b))
    left_part = (centre - a) / 6.0 * (f(a) + 4.0 * f(0.5 * (a + centre)) + f(centre))
    right_part = (b - centre) / 6.0 * (f(centre) + 4.0 * f(0.5 * (centre + b)) + f(b))
    if depth > 40 or abs(left_part + right_part - whole) <= 15.0 * tol:
        return left_part + right_part + (left_part + right_part - whole) / 15.0
    return (adaptive_simpson(f, a, centre, tol / 2.0, depth + 1)
            + adaptive_simpson(f, centre, b, tol / 2.0, depth + 1))


def run_task_3_2() -> None:
    """Задание 3.2 (п. 6.5): вычисление интеграла формулой Гаусса."""
    section("ЗАДАНИЕ 3.2. ИНТЕГРИРОВАНИЕ ФОРМУЛОЙ ГАУССА")
    print(f"  Интеграл варианта 10: I = integral from {INTEGRAL_A} to {INTEGRAL_B} "
          f"of exp(-x^2)*cos(x) dx")
    print(f"  Порядок формулы по варианту: n = {INTEGRAL_N_ORDER}")
    print("")
    print("  Входные данные: порядок формулы, границы интервала,")
    print("  подынтегральная функция, относительная точность.")

    order = read_int("  Порядок формулы Гаусса n", INTEGRAL_N_ORDER, 1, 20)
    rel_tol = read_float("  Относительная точность eps", 1e-10)
    if rel_tol <= 0.0:
        print("  Точность заменена на значение по умолчанию 1e-10.")
        rel_tol = 1e-10

    started = time.perf_counter()
    a, b = INTEGRAL_A, INTEGRAL_B

    nodes, weights = gauss_nodes_weights(order)
    print("")
    print(f"  Этап 1. Узлы и веса формулы Гаусса порядка n = {order} на [-1; 1]")
    rows = []
    for i, (t_i, w_i) in enumerate(zip(nodes, weights)):
        check = legendre_polynomial(order, t_i)
        rows.append((str(i), f"{t_i:.14f}", f"{w_i:.14f}", f"{check:.3e}"))
    print_table(("i", "t_i", "w_i", "P_n(t_i)"), rows)
    print(f"  Сумма весов: {sum(weights):.14f} (должна быть равна 2)")

    # автоматический выбор шага: удвоение числа подотрезков, пока
    # поправка Ричардсона |I_2N - I_N|/(2^(2n) - 1) не станет меньше eps
    divisor = float(2 ** (2 * order) - 1)
    parts = 1
    value_prev = gauss_integral(integrand, a, b, order, parts)
    # первая строка журнала: результат на одном подотрезке (оценки ещё нет)
    history: List[Tuple[int, float, float, float]] = [(parts, value_prev, float("nan"), float("nan"))]
    value_curr = value_prev
    parts_curr = parts
    while parts < 1 << 20:
        parts *= 2
        value_curr = gauss_integral(integrand, a, b, order, parts)
        estimate = abs(value_curr - value_prev) / divisor
        relative = estimate / max(abs(value_curr), 1e-300)
        history.append((parts, value_curr, estimate, relative))
        value_prev = value_curr
        parts_curr = parts
        if relative <= rel_tol:
            break

    elapsed = (time.perf_counter() - started) * 1e3

    print("")
    print("  Этап 2. Автоматический выбор шага интегрирования")
    print(f"  Поправка Ричардсона для формулы степени 2n: |I_2N - I_N| / (2^(2n) - 1),"
          f"  делитель = {divisor:.0f}")
    rows = []
    for n_int, val, est, rel in history:
        mark = "*" if rel <= rel_tol else ""
        est_text = "  --    " if math.isnan(est) else f"{est:.3e}"
        rel_text = "  --    " if math.isnan(rel) else f"{rel:.3e}"
        rows.append((str(n_int), f"{val:.14f}", est_text, rel_text, mark))
    print_table(("N", "I_N", "поправка", "отн. оценка", "готово"), rows)
    print("  Первая строка - результат на одном подотрезке (N = 1);")

    reference = adaptive_simpson(integrand, a, b, 1e-14)
    achieved = abs(value_curr - reference) / max(abs(reference), 1e-300)

    print("")
    print("  Этап 3. Итоговые результаты")
    print(f"    Значение интеграла I            = {value_curr:.14f}")
    print(f"    Число подотрезков N             = {parts_curr}")
    print(f"    Количество узлов Гаусса всего   = {parts_curr * order}")
    print(f"    Оценка точности (Ричардсон)     = {history[-1][2]:.3e}")
    print(f"    Достигнутая относительная погрешность = {achieved:.3e}")
    print(f"    Заданная относительная точность       = {rel_tol:.3e}")
    verdict = "выполнена" if achieved <= max(rel_tol, 1e-14) else "не выполнена"
    print(f"    Требование точности: {verdict}")
    print(f"    Контрольное значение (адаптивный Симпсон) = {reference:.14f}")
    print(f"    Время выполнения: {elapsed:.3f} мс")


# ===========================================================================
#  МЕНЮ ВЫБОРА ЗАДАНИЯ
# ===========================================================================

MENU = """
------------------------------------------------------------------------------
  1 - Задание 1.   Интерполяция полиномом Ньютона (п. 6.3)
  2 - Задание 2.   Численное дифференцирование (п. 6.4)
  3 - Задание 3.1. Формула трапеций, автовыбор шага (п. 6.5)
  4 - Задание 3.2. Формула Гаусса (п. 6.5)
  0 - Выход
------------------------------------------------------------------------------
"""

TASKS = {
    "1": run_task_1,
    "2": run_task_2,
    "3": run_task_3_1,
    "4": run_task_3_2,
}


def main() -> None:
    """Главная процедура: цикл меню выбора задания."""
    print("=" * 78)
    print("Лабораторная работа № 2 по дисциплине «Вычислительная математика»")
    print("Интерполяция, численное дифференцирование и интегрирование")
    print("Вариант 10")
    print("=" * 78)

    while True:
        print(MENU)
        choice = _input("Выберите задание: ").strip()
        if choice == "0":
            print("Работа программы завершена.")
            return
        task = TASKS.get(choice)
        if task is None:
            print("Нет такого пункта меню. Повторите выбор.")
            continue
        try:
            task()
        except ValueError as error:
            print(f"\nОшибка входных данных: {error}")
        print("")
        _input("Нажмите Enter для возврата в меню...")


if __name__ == "__main__":
    main()
