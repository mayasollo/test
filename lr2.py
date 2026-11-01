#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа № 2
Дисциплина: «Вычислительная математика», вариант 10.

Задание 1 (п. 6.3). Интерполяция таблично заданной функции
    y = sin^2(x) + 1 на [0; pi/2] с шагом h = pi/20.
    По узлам исходной сетки строится полином Ньютона в разделённых
    разностях; значения вычисляются на новой сетке x_j = j*h/2.

Задание 2 (п. 6.4). Численное дифференцирование.
    Первая и вторая производные находятся дифференцированием
    полинома Ньютона; сравниваются с точными y' = sin(2x) и
    y'' = 2*cos(2x).

Задание 3.1 (п. 6.5). Интеграл по формуле трапеций
    I = integral(exp(-x^2)*cos(x), 0, 2); шаг выбирается автоматически
    из условия |I_2n - I_n|/3 <= eps*|I_2n|.

Задание 3.2 (п. 6.5). Тот же интеграл по формуле Гаусса n = 6;
    узлы и веса вычисляются через многочлены Лежандра.

Используется только стандартный модуль math.

Запуск:  python3 lr2.py
"""

import math


# --- данные варианта 10 -----------------------------------------------------
H_STEP = math.pi / 20.0
OLD_GRID = [i * H_STEP for i in range(11)]          # x_i = i*h, 11 узлов
NEW_GRID = [j * H_STEP / 2.0 for j in range(21)]    # x_j = j*h/2, 21 узел

INT_A = 0.0
INT_B = 2.0
GAUSS_N = 6


def f_tab(x):
    """Интерполируемая функция y = sin^2(x) + 1."""
    s = math.sin(x)
    return s * s + 1.0


def df_true(x):
    """Точная первая производная y' = sin(2x)."""
    return math.sin(2.0 * x)


def ddf_true(x):
    """Точная вторая производная y'' = 2*cos(2x)."""
    return 2.0 * math.cos(2.0 * x)


def f_integrand(x):
    """Подынтегральная функция exp(-x^2)*cos(x)."""
    return math.exp(-x * x) * math.cos(x)


def ask_number(text, default):
    while True:
        try:
            raw = input(f"{text} [{default}]: ").strip()
        except EOFError:
            raise SystemExit("\nВвод прерван, работа завершена.")
        if raw == "":
            return float(default)
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            print("  Нужно ввести число.")


def ask_int(text, default, low, high):
    while True:
        try:
            raw = input(f"{text} [{default}]: ").strip()
        except EOFError:
            raise SystemExit("\nВвод прерван, работа завершена.")
        if raw == "":
            return int(default)
        try:
            value = int(raw)
            if low <= value <= high:
                return value
        except ValueError:
            pass
        print(f"  Нужно целое число от {low} до {high}.")


def split_diffs(xs, ys):
    """Разделённые разности f[x_0], f[x_0,x_1], ..., f[x_0..x_m].

    Массив пересчитывается на месте по столбцам треугольной схемы.
    """
    c = list(ys)
    m = len(xs)
    for k in range(1, m):
        for i in range(m - 1, k - 1, -1):
            c[i] = (c[i] - c[i - 1]) / (xs[i] - xs[i - k])
    return c


def pick_nodes(x, count):
    """Возвращает count ближайших к точке x узлов исходной сетки."""
    ordered = sorted(OLD_GRID, key=lambda t: abs(t - x))
    chosen = ordered[:count]
    return sorted(chosen)


def newton_all(xs, c, x):
    """Полином Ньютона и его две производные в точке x.

    Идут базисные множители w_k = (x-x_0)...(x-x_{k-1});
    их производные получаются рекуррентно при том же проходе.
    """
    p = c[0]
    dp = 0.0
    ddp = 0.0
    w = 1.0        # значение базисного многочлена w_k
    dw = 0.0       # w'_k
    ddw = 0.0      # w''_k
    for k in range(1, len(c)):
        t = x - xs[k - 1]
        ddw = ddw * t + 2.0 * dw
        dw = dw * t + w
        w = w * t
        p += c[k] * w
        dp += c[k] * dw
        ddp += c[k] * ddw
    return p, dp, ddp


def omega_value(xs, x):
    """Узловой многочлен (x-x_0)(x-x_1)...(x-x_m)."""
    prod = 1.0
    for t in xs:
        prod *= (x - t)
    return prod


def task_interpolation():
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 1. ИНТЕРПОЛЯЦИЯ ПОЛИНОМОМ НЬЮТОНА")
    print("=" * 70)
    print("y = sin^2(x) + 1, [0; pi/2], h = pi/20; новая сетка x_j = j*h/2")

    m = ask_int("Порядок полинома Ньютона m", 3, 1, 10)

    print("\nИсходная таблица:")
    print(f"{'i':>3} {'x_i':>12} {'y_i':>12}")
    for i, t in enumerate(OLD_GRID):
        print(f"{i:>3} {t:>12.8f} {f_tab(t):>12.8f}")

    print(f"\nРасчёт на новой сетке (m = {m}):")
    print(f"{'j':>3} {'x_j':>12} {'y(x_j)':>12} {'P_m(x_j)':>12} "
          f"{'|y-P_m|':>12} {'оценка':>12}")
    worst_real = 0.0
    worst_bound = 0.0
    fact = 1.0
    for q in range(1, m + 2):
        fact *= q                       # fact = (m+1)!
    bound_coef = 2.0 ** m / fact        # M_{m+1}/(m+1)!,  M_{m+1} = 2^m

    for j, x in enumerate(NEW_GRID):
        xs = pick_nodes(x, m + 1)
        c = split_diffs(xs, [f_tab(t) for t in xs])
        p, _, _ = newton_all(xs, c, x)
        real = abs(f_tab(x) - p)
        est = bound_coef * abs(omega_value(xs, x))
        worst_real = max(worst_real, real)
        worst_bound = max(worst_bound, est)
        print(f"{j:>3} {x:>12.8f} {f_tab(x):>12.8f} {p:>12.8f} "
              f"{real:>12.4e} {est:>12.4e}")

    ok = "подтверждается" if worst_real <= worst_bound * 1.000001 else "НАРУШЕНА"
    print(f"\nМаксимальная фактическая погрешность: {worst_real:.6e}")
    print(f"Максимальная теоретическая оценка:    {worst_bound:.6e}")
    print(f"Неравенство |y - P_m| <= M_(m+1)/(m+1)! * |omega| {ok}.")


def task_differentiation():
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 2. ЧИСЛЕННОЕ ДИФФЕРЕНЦИРОВАНИЕ")
    print("=" * 70)

    m = ask_int("Порядок полинома Ньютона m", 3, 1, 10)

    print(f"\nПервая производная (m = {m}):")
    print(f"{'j':>3} {'x_j':>12} {'y_true':>12} {'Pm_shtrih':>12} {'otl':>12}")
    e1 = 0.0
    for j, x in enumerate(NEW_GRID):
        xs = pick_nodes(x, m + 1)
        c = split_diffs(xs, [f_tab(t) for t in xs])
        _, dp, _ = newton_all(xs, c, x)
        err = abs(df_true(x) - dp)
        e1 = max(e1, err)
        print(f"{j:>3} {x:>12.8f} {df_true(x):>12.8f} {dp:>12.8f} {err:>12.4e}")

    print(f"\nВторая производная (m = {m}):")
    print(f"{'j':>3} {'x_j':>12} {'y_true_dd':>12} {'Pm_ddv':>12} {'otl':>12}")
    e2 = 0.0
    for j, x in enumerate(NEW_GRID):
        xs = pick_nodes(x, m + 1)
        c = split_diffs(xs, [f_tab(t) for t in xs])
        _, _, ddp = newton_all(xs, c, x)
        err = abs(ddf_true(x) - ddp)
        e2 = max(e2, err)
        print(f"{j:>3} {x:>12.8f} {ddf_true(x):>12.8f} {ddp:>12.8f} {err:>12.4e}")

    print(f"\nМаксимальная погрешность первой производной:  {e1:.6e}")
    print(f"Максимальная погрешность второй производной:  {e2:.6e}")


def trapezoid(a, b, parts):
    """Составная формула трапеций на равных шагах."""
    h = (b - a) / parts
    total = 0.5 * (f_integrand(a) + f_integrand(b))
    i = 1
    while i < parts:
        total += f_integrand(a + i * h)
        i += 1
    return total * h


def task_trapezoid():
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 3.1. ФОРМУЛА ТРАПЕЦИЙ, АВТОМАТИЧЕСКИЙ ВЫБОР ШАГА")
    print("=" * 70)
    print(f"I = integral(exp(-x^2)*cos(x), {INT_A}, {INT_B})")

    eps = ask_number("Требуемая относительная точность eps", 1e-6)
    parts = 4                                 # начальное число интервалов

    prev = trapezoid(INT_A, INT_B, parts)
    print(f"\n{'n':>8} {'I_n':>16} {'оценка |dI|':>12} {'относительная':>14}")
    print(f"{parts:>8} {prev:>16.12f} {'-':>12} {'-':>14}")
    while True:
        parts *= 2
        curr = trapezoid(INT_A, INT_B, parts)
        gap = abs(curr - prev) / 3.0          # оценка по Ричардсону
        rel = gap / abs(curr)
        print(f"{parts:>8} {curr:>16.12f} {gap:>12.4e} {rel:>14.4e}")
        if rel <= eps:
            break
        prev = curr
        if parts > 4000000:
            print("Достигнут предел числа интервалов.")
            break

    print(f"\nИтог: I = {curr:.12f} при числе интервалов n = {parts}, "
          f"шаг h = {(INT_B - INT_A) / parts:.6e}")
    print(f"Оценённая относительная погрешность {rel:.4e} <= eps = {eps:.4e}.")


def legendre(n, t):
    """Многочлен Лежандра P_n(t) по рекуррентному соотношению."""
    p_prev, p_cur = 1.0, t
    if n == 0:
        return 1.0
    for k in range(1, n):
        p_prev, p_cur = p_cur, ((2 * k + 1) * t * p_cur - k * p_prev) / (k + 1)
    return p_cur


def legendre_der(n, t):
    """Производная P_n'(t) = n*(t*P_n - P_{n-1})/(t^2 - 1)."""
    if n == 0:
        return 0.0
    num = n * (t * legendre(n, t) - legendre(n - 1, t))
    den = t * t - 1.0
    if abs(den) < 1e-14:                      # особые точки t = +-1
        return n * (n + 1) / 2.0 * (1.0 if t > 0 else (-1.0) ** n)
    return num / den


def gauss_table(n):
    """Корни P_n(t) методом Ньютона и соответствующие веса формулы Гаусса."""
    nodes, weights = [], []
    for i in range(n):
        t = math.cos(math.pi * (i + 0.75) / (n + 0.5))
        for _ in range(60):
            step = legendre(n, t) / legendre_der(n, t)
            t -= step
            if abs(step) < 1e-16:
                break
        nodes.append(t)
        d = legendre_der(n, t)
        weights.append(2.0 / ((1.0 - t * t) * d * d))
    return nodes, weights


def gauss_sum(nodes, weights, a, b, parts):
    """Составная формула Гаусса: [a, b] разбит на parts равных частей."""
    width = (b - a) / parts
    total = 0.0
    for k in range(parts):
        left = a + k * width
        middle = left + width / 2.0
        half = width / 2.0
        for t, w in zip(nodes, weights):
            total += w * f_integrand(middle + half * t)
    return total * half


def task_gauss():
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 3.2. ФОРМУЛА ГАУССА")
    print("=" * 70)

    n = ask_int("Порядок формулы Гаусса", GAUSS_N, 1, 12)
    eps = ask_number("Требуемая относительная точность eps", 1e-10)

    nodes, weights = gauss_table(n)
    print(f"\nУзлы и веса формулы Гаусса {n}-го порядка на [-1; 1]:")
    print(f"{'i':>3} {'t_i':>16} {'w_i':>16} {'P_n(t_i)':>12}")
    for i, (t, w) in enumerate(zip(nodes, weights)):
        print(f"{i:>3} {t:>16.14f} {w:>16.14f} {legendre(n, t):>12.3e}")
    print(f"Сумма весов = {sum(weights):.14f} (контроль: должна быть 2)")

    divisor = float(2 ** (2 * n) - 1)         # знаменатель поправки Ричардсона
    parts = 1
    prev = gauss_sum(nodes, weights, INT_A, INT_B, parts)
    print(f"\n{'N':>6} {'I_N':>18} {'поправка':>12} {'относительная':>14}")
    print(f"{parts:>6} {prev:>18.14f} {'-':>12} {'-':>14}")
    while True:
        parts *= 2
        curr = gauss_sum(nodes, weights, INT_A, INT_B, parts)
        gap = abs(curr - prev) / divisor
        rel = gap / abs(curr)
        print(f"{parts:>6} {curr:>18.14f} {gap:>12.4e} {rel:>14.4e}")
        if rel <= eps or parts > 200000:
            break
        prev = curr

    print(f"\nИтог: I = {curr:.14f}, порядок формулы n = {n}, "
          f"число подотрезков N = {parts}")
    print(f"Оценённая относительная погрешность {rel:.4e} <= eps = {eps:.4e}.")


MENU = """
======================================================================
 Лабораторная работа № 2, вариант 10
======================================================================
  1 - Задание 1.   Интерполяция полиномом Ньютона
  2 - Задание 2.   Численное дифференцирование
  3 - Задание 3.1. Формула трапеций (автовыбор шага)
  4 - Задание 3.2. Формула Гаусса
  0 - Выход
"""

TASKS = {"1": task_interpolation, "2": task_differentiation,
         "3": task_trapezoid, "4": task_gauss}


def main():
    while True:
        print(MENU)
        try:
            choice = input("Выберите задание: ").strip()
        except EOFError:
            print("\nРабота завершена.")
            return
        if choice == "0":
            print("Работа завершена.")
            return
        task = TASKS.get(choice)
        if task is None:
            print("Нет такого пункта меню.")
            continue
        task()
        try:
            input("\nНажмите Enter для возврата в меню...")
        except EOFError:
            print("\nРабота завершена.")
            return


if __name__ == "__main__":
    main()
