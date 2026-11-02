#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа № 2. Вариант 10.

Задание 1 (п. 6.3). Интерполяция таблично заданной функции
    y = sin^2(x) + 1 на [0; pi/2], шаг h = pi/20, новая сетка x_j = j*h/2,
    полиномом Ньютона (разделённые разности).
Задание 2 (п. 6.4). Первая и вторая производные той же функции -
    дифференцированием полинома Ньютона.
Задание 3.1 (п. 6.5). Интеграл варианта I = int_0^2 exp(-x^2)*cos(x) dx
    по формуле трапеций с автоматическим выбором шага (правило Рунге).
Задание 3.2 (п. 6.5). Тот же интеграл по формуле Гаусса, n = 6
    (узлы - корни полинома Лежандра, веса - из системы или формулы).

Алгоритмы реализованы самостоятельно по формулам учебного пособия
(Мицель А.А. «Вычислительные методы», гл. 3-5) и методических указаний.

Запуск:  python3 lr2.py
"""
import math

PI = math.pi


def f_true(x):
    """Функция варианта 10: y = sin^2(x) + 1."""
    s = math.sin(x)
    return s * s + 1.0


def df_true(x):
    """y' = sin(2x)."""
    return math.sin(2.0 * x)


def ddf_true(x):
    """y'' = 2*cos(2x)."""
    return 2.0 * math.cos(2.0 * x)


def make_grid():
    """Исходная сетка x_i = i*pi/20, i = 0..10; новая x_j = j*pi/40, j = 0..20."""
    xs = [i * PI / 20.0 for i in range(11)]
    ys = [f_true(x) for x in xs]
    xn = [j * PI / 40.0 for j in range(21)]
    return xs, ys, xn


def window(xs, ys, x, m):
    """Номера m+1 ближайших узлов исходной сетки под точку x
    (окно сдвигается к краю, если точка близка к границе)."""
    n = len(xs)
    i = max(0, min(n - m - 1, int((x - xs[0]) / (xs[1] - xs[0])) - (m - 1) // 2))
    return xs[i:i + m + 1], ys[i:i + m + 1]


def poly_newton(wx, wy, x, der=0):
    """Полином Ньютона степени m=len(wx)-1 и его производная порядка der.
    Считается рекуррентно по базисным многочленам w_k(x)=(x-x_0)...(x-x_{k-1}):
        P(x)   = sum_k c_k * w_k(x),      c_k = f[x_0..x_k]
    Для производных применяется правило Лейбница к w_k.
    Возвращает (P, P', P'') при der=2."""
    m = len(wx) - 1
    dd = diff_table_from(wx, wy, m)
    # значения базисных многочленов и их производных до 2-го порядка
    wv = [1.0, 0.0, 0.0]           # w_0 = 1: (значение, w', w'')
    p = [0.0, 0.0, 0.0]
    for k in range(m + 1):
        if k > 0:
            # w_k = w_{k-1} * (x - x_{k-1})  ->  правило Лейбница
            a = x - wx[k - 1]
            nw0 = wv[0] * a
            nw1 = wv[1] * a + wv[0]
            nw2 = wv[2] * a + 2.0 * wv[1]
            wv = [nw0, nw1, nw2]
        c = dd[0][k]
        for t in range(3):
            p[t] += c * wv[t]
    return p[der]


def diff_table_from(xs, ys, m):
    """Разделённые разности для произвольного окна (см. diff_table)."""
    dd = [[0.0] * (m + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dd[i][0] = ys[i]
    for k in range(1, m + 1):
        for i in range(m + 1 - k):
            dd[i][k] = (dd[i + 1][k - 1] - dd[i][k - 1]) / (xs[i + k] - xs[i])
    return dd


def node_poly_derivs(wx, x):
    """omega(x) = (x-x_0)...(x-x_m) и его логарифмические производные:
    возвращает (omega, omega', omega'').
    omega'  = omega * sum 1/(x-x_i)
    omega'' = omega * ((sum 1/(x-x_i))^2 - sum 1/(x-x_i)^2)
    (пособие, ф-лы после 4.13)."""
    om = 1.0
    for xi in wx:
        om *= (x - xi)
    s1 = 0.0
    s2 = 0.0
    near = False
    for xi in wx:
        d = x - xi
        if abs(d) < 1e-12:
            near = True
        else:
            s1 += 1.0 / d
            s2 += 1.0 / (d * d)
    if near:
        # в узле omega=0; производные считаем прямым суммированием
        o1 = 0.0
        o2 = 0.0
        mnum = len(wx)
        for i in range(mnum):
            di = 1.0
            for j in range(mnum):
                if j != i:
                    di *= (x - wx[j])
            o1 += di
            for j in range(i + 1, mnum):
                dij = 1.0
                for l in range(mnum):
                    if l != i and l != j:
                        dij *= (x - wx[l])
                o2 += 2.0 * dij
        return 0.0, o1, o2
    return om, om * s1, om * (s1 * s1 - s2)


def bound_deriv(k):
    """M_k = max |y^(k)(x)| на [0; pi/2] для y = sin^2 x + 1 = (1-cos2x)/2+1.
    y^(k) = 2^{k-1}*trig(2x), поэтому M_k = 2^{k-1}, k>=1."""
    return 2.0 ** (k - 1)


def task1():
    print("=" * 72)
    print("ЗАДАНИЕ 1. ИНТЕРПОЛЯЦИЯ ПОЛИНОМОМ НЬЮТОНА (вариант 10)")
    print("y = sin^2(x)+1, x_i = i*pi/20 (i=0..10); новая сетка x_j = j*pi/40")
    print("=" * 72)
    xs, ys, xn = make_grid()
    m = 3
    print("\nИсходная сетка:")
    for i in range(len(xs)):
        print("  x_%2d = %9.6f   y = %10.6f" % (i, xs[i], ys[i]))
    print("\nНовая сетка (m = %d, окно %d ближайших узлов):" % (m, m + 1))
    print("   j     x_j        P_m(x_j)      y(x_j)       |y-P|      оценка R_m")
    emax = 0.0
    for j, x in enumerate(xn):
        wx, wy = window(xs, ys, x, m)
        p = poly_newton(wx, wy, x, 0)
        # оценка погрешности (3.3): |R| <= M_{m+1}/(m+1)! * |omega(x)|
        om, _, _ = node_poly_derivs(wx, x)
        est = abs(om) * bound_deriv(m + 1) / math.factorial(m + 1)
        err = abs(f_true(x) - p)
        emax = max(emax, err)
        print("  %3d  %9.6f  %12.8f  %12.8f  %10.2e  %10.2e"
              % (j, x, p, f_true(x), err, est))
    print("\nМаксимальная фактическая погрешность: %.3e" % emax)


def task2():
    print("=" * 72)
    print("ЗАДАНИЕ 2. ЧИСЛЕННОЕ ДИФФЕРЕНЦИРОВАНИЕ (полином Ньютона, m=3)")
    print("=" * 72)
    xs, ys, xn = make_grid()
    m = 3
    print("   j     x_j       P'_m(x_j)     y'(x_j)      eps1     "
          "P''_m(x_j)     y''(x_j)     eps2")
    e1 = 0.0
    e2 = 0.0
    for j, x in enumerate(xn):
        wx, wy = window(xs, ys, x, m)
        p1 = poly_newton(wx, wy, x, 1)
        p2 = poly_newton(wx, wy, x, 2)
        r1 = abs(p1 - df_true(x))
        r2 = abs(p2 - ddf_true(x))
        e1 = max(e1, r1)
        e2 = max(e2, r2)
        print("  %3d  %9.6f  %11.7f  %11.7f  %8.2e  %11.7f  %11.7f  %8.2e"
              % (j, x, p1, df_true(x), r1, p2, ddf_true(x), r2))
    # оценки погрешностей (пособие, ф-лы после 4.13):
    # |R'| <= M_{m+1}/(m+1)! * |omega'|,  |R''| <= M_{m+1}/(m+1)! * |omega''|
    print("\nОценки сверху (M_4=8, |omega'|, |omega''| - максимум по сетке):")
    mo1 = 0.0
    mo2 = 0.0
    for x in xn:
        wx, _ = window(xs, ys, x, m)
        _, o1, o2 = node_poly_derivs(wx, x)
        mo1 = max(mo1, abs(o1))
        mo2 = max(mo2, abs(o2))
    fac = math.factorial(m + 1)
    print("  max|R'| ~ M_4/(4!) * max|omega'| = %.2e" % (8.0 / fac * mo1))
    print("  max|R''|~ M_4/(4!) * max|omega''|= %.2e" % (8.0 / fac * mo2))
    print("Фактически: max eps1 = %.2e, max eps2 = %.2e" % (e1, e2))


def integrand(x):
    """Подынтегральная функция задания 3 варианта 10:
    I = int_0^2 e^{-x^2} * cos(x) dx."""
    return math.exp(-x * x) * math.cos(x)


def trapezoid(a, b, n):
    """Сумма трапеций на n равных интервалах (пособие, ф-ла 5.4)."""
    h = (b - a) / n
    s = 0.5 * (integrand(a) + integrand(b))
    for i in range(1, n):
        s += integrand(a + i * h)
    return s * h


def task31():
    print("=" * 72)
    print("ЗАДАНИЕ 3.1. ФОРМУЛА ТРАПЕЦИЙ С АВТОМАТИЧЕСКИМ ВЫБОРОМ ШАГА")
    print("I = int_0^2 e^{-x^2} cos(x) dx,  правило Рунге (ф-ла 5.16), m=2")
    print("=" * 72)
    a = 0.0
    b = 2.0
    delta = 1e-6
    n = 8
    prev = trapezoid(a, b, n)
    print("   n        S_n            оценка Рунге      отн. ошибка")
    while True:
        n *= 2
        cur = trapezoid(a, b, n)
        est = abs(cur - prev) / (2.0 ** 2 - 1.0)   # Рунге для трапеций
        rel = est / abs(cur)
        print("  %4d  %16.10f    %12.4e    %12.4e" % (n, cur, est, rel))
        if rel <= delta:
            break
        prev = cur
    print("\nИтог: I = %.10f при n = %d; оценка погрешности %.3e (delta=%.0e)"
          % (cur, n, est, delta))
    exact = 0.6926114516163757   # контрольное значение (высокая точность)
    print("Контроль: |I - I_точн| = %.3e" % abs(cur - exact))


def legendre(n, t):
    """Полином Лежандра P_n(t) по рекуррентной формуле (пособие, 3.17)."""
    p0 = 1.0
    if n == 0:
        return p0
    p1 = t
    for k in range(1, n):
        p2 = ((2.0 * k + 1.0) * t * p1 - k * p0) / (k + 1.0)
        p0, p1 = p1, p2
    return p1


def legendre_d(n, t):
    """Производная P_n'(t) = (n*t*P_n(t) - n*P_{n-1}(t)) / (t^2 - 1)."""
    if n == 0:
        return 0.0
    return (n * t * legendre(n, t) - n * legendre(n - 1, t)) / (t * t - 1.0)


def find_root_poly_between(n, lo, hi):
    """Корень P_n на [lo, hi]: бисекция знака + ускорение Ньютоном.
    Ньютон применяется, если шаг остаётся внутри отрезка, иначе — деление
    пополам; гарантированно сходится, т.к. корни P_n простые."""
    flo = legendre(n, lo)
    for _ in range(100):
        fmid = legendre(n, 0.5 * (lo + hi))
        x = 0.5 * (lo + hi)
        gx = legendre_d(n, x)
        if gx != 0.0:
            cand = x - fmid / gx
            if lo < cand < hi:
                x = cand
        fx = legendre(n, x)
        if abs(hi - lo) < 1e-15 or fx == 0.0:
            return x
        if flo * fx <= 0.0:
            hi = x
        else:
            lo, flo = x, fx
    return 0.5 * (lo + hi)


def gauss_nodes_weights(n):
    """Узлы t_i - все корни P_n(t) на (-1,1); веса
    A_i = 2 / ((1 - t_i^2) * (P_n'(t_i))^2)   (пособие, п. 5.4).
    Корни ищутся перебором подинтервалов сетки с проверкой смены знака."""
    g = 400
    grid = [-1.0 + 2.0 * i / g for i in range(g + 1)]
    ts = []
    for i in range(g):
        u, v = grid[i], grid[i + 1]
        fu, fv = legendre(n, u), legendre(n, v)
        if fu == 0.0:
            ts.append(u)
        elif fu * fv < 0.0:
            ts.append(find_root_poly_between(n, u, v))
    ts.sort()
    ws = [2.0 / ((1.0 - t * t) * legendre_d(n, t) ** 2) for t in ts]
    return ts, ws


def task32():
    print("=" * 72)
    print("ЗАДАНИЕ 3.2. КВАДРАТУРНАЯ ФОРМУЛА ГАУССА (n = 6, вариант 10)")
    print("I = int_0^2 e^{-x^2} cos(x) dx")
    print("=" * 72)
    n = 6
    ts, ws = gauss_nodes_weights(n)
    print("\nУзлы t_i и веса A_i формулы Гаусса (расчётные):")
    print("  эталон (таблица 5.1 пособия): +-0.93246951, +-0.66120939, +-0.23861919")
    print("                                0.17132450,  0.36076158,  0.46791394")
    for i in range(n):
        print("  t_%d = %+11.8f   A_%d = %10.8f" % (i + 1, ts[i], i + 1, ws[i]))
    sws = sum(ws)
    print("Контроль: сумма весов = %.12f (должна быть равна 2)" % sws)

    a = 0.0
    b = 2.0
    xc = 0.5 * (a + b)
    hl = 0.5 * (b - a)
    delta = 1e-8
    jmax = 0
    prev = None
    s = 0.0
    parts = 1
    print("\n   j    S_j (ф-ла 5.20)         оценка (S_j-S_{j-1})/S_j")
    while True:
        parts = jmax + 1
        s = 0.0
        for q in range(parts):
            l = a + (b - a) * q / parts
            r = a + (b - a) * (q + 1) / parts
            cx = 0.5 * (l + r)
            hx = 0.5 * (r - l)
            sub = 0.0
            for i in range(n):
                sub += ws[i] * integrand(cx + hx * ts[i])
            s += hx * sub
        if prev is not None:
            rel = abs(s - prev) / abs(s)
            print("  %3d    %16.12f      %12.4e" % (jmax, s, rel))
            if rel <= delta:
                break
        prev = s
        jmax += 1
    print("\nИтог: I = %.12f (подынтервалов: %d, узлов всего: %d)"
          % (s, parts, parts * n))
    exact = 0.6926114516163757
    print("Контроль: |I - I_точн| = %.3e" % abs(s - exact))


MENU = """
======================================================================
 ЛР N2, вариант 10
   1 - Интерполяция (Ньютон)
   2 - Численное дифференцирование
   3 - Интегрирование: трапеции, автошаг (Рунге)
   4 - Интегрирование: формула Гаусса (n=6)
   0 - выход
======================================================================"""


def ask(prompt):
    try:
        return input(prompt)
    except EOFError:
        raise SystemExit("\nВвод прерван, программа завершена.")


if __name__ == "__main__":
    while True:
        print(MENU)
        c = ask("Выбор: ").strip()
        if c == "1":
            task1()
        elif c == "2":
            task2()
        elif c == "3":
            task31()
        elif c == "4":
            task32()
        elif c == "0":
            break
        else:
            print("Нет такого пункта.")
