#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа № 1. Вариант 10.

Задание 1. Отделение корней уравнения f(x) = 10*cos(x) - 0.1*x^2 = 0
           и их уточнение методом Ньютона.
Задание 2. Метод Гаусса: решение СЛАУ, определитель, обратная матрица.

Реализация написана самостоятельно, по формулам методических указаний
(п. 1.2-1.4, 2.1, 2.3, 2.4).

Запуск:  python3 lr1.py
"""
import math
import time

# ----- данные варианта 10 ----------------------------------------------------
A_LEFT = -10.0          # левая граница отрезка [a, b] (из задания)
B_RIGHT = 10.0          # правая граница отрезка
H_STEP = 0.25           # фиксированный шаг сетки отделения; пользователь
                        # его не вводит (в перечне входных данных его нет).
                        # Обоснование: f'(x) = -10*sin(x) - 0.2*x, нули f'
                        # (экстремумы) расположены примерно через pi, т.е.
                        # наименьшая "ямка" монотонности имеет длину ~3.14.
                        # Шаг h = 0.25 даёт >= 12 узлов между соседними
                        # экстремумами, что надёжно отделяет все корни, а
                        # дальнейшее уменьшение шага уже не меняет результат
                        # (проверено при h = 0.1 и h = 0.05).
EPS1 = 1e-8             # точность по аргументу |x_k - x_(k-1)|
EPS2 = 1e-8             # точность по функции |f(x_k)|

MAT_A = [[7.9, 5.6, 5.7, -7.2],
         [8.5, -4.8, 0.8, 3.5],
         [4.3, 4.2, -3.2, 9.3],
         [3.2, -1.4, -8.9, 3.3]]
VEC_B = [6.68, 9.95, 8.6, 1.0]


def func(x):
    """f(x) = 10*cos(x) - 0.1*x^2."""
    return 10.0 * math.cos(x) - 0.1 * x * x


def d1f(x):
    """f'(x) = -10*sin(x) - 0.2*x."""
    return -10.0 * math.sin(x) - 0.2 * x


def d2f(x):
    """f''(x) = -10*cos(x) - 0.2."""
    return -10.0 * math.cos(x) - 0.2


def isolate(a, b, h):
    """Отделение корней на равномерной сетке с шагом h.

    Возвращает список пар (x_i, x_{i+1}) - подотрезков сетки, на которых
    f меняет знак (f(x_i)*f(x_{i+1}) < 0). По построению шага (см. H_STEP)
    каждый такой подотрезок лежит внутри одной области монотонности f,
    поэтому содержит ровно один корень.
    """
    segs = []
    x1 = a
    f1 = func(x1)
    n = int(math.ceil((b - a) / h))
    for i in range(1, n + 1):
        x2 = a + i * h
        if x2 > b:
            x2 = b
        f2 = func(x2)
        if f1 == 0.0:
            segs.append((x1, x1))
        elif f1 * f2 < 0.0:
            segs.append((x1, x2))
        x1, f1 = x2, f2
    if f1 == 0.0:
        segs.append((x1, x1))
    return segs


def pick_start(a, b):
    """Начальное приближение метода Ньютона: тот конец отрезка, для которого
    f(x)*f''(x) > 0 (условие сходимости касательных); иначе - середина."""
    if func(a) * d2f(a) > 0.0:
        return a, "конец a (f*f'' > 0)"
    if func(b) * d2f(b) > 0.0:
        return b, "конец b (f*f'' > 0)"
    return 0.5 * (a + b), "середина (условие не выполнено ни на одном конце)"


def newton(x0, eps1, eps2):
    """Метод Ньютона: x_k = x_(k-1) - f(x_(k-1))/f'(x_(k-1)).

    Останов: |x_k - x_(k-1)| < eps1 и |f(x_k)| < eps2.
    Возвращает (корень, история [(x_k, f(x_k))...]).
    """
    hist = [(x0, func(x0))]
    x = x0
    for _ in range(100):
        d = d1f(x)
        if abs(d) < 1e-14:
            break
        xn = x - func(x) / d
        fn = func(xn)
        hist.append((xn, fn))
        if abs(xn - x) < eps1 and abs(fn) < eps2:
            break
        x = xn
    return hist[-1][0], hist


def alpha_of(hist):
    """Фактический параметр сходимости (k=2): alpha = D_k / D_(k-1)^2,
    D_k = |x_k - x_(k-1)|. Берётся последняя тройка точек, ещё не искажённая
    машинным округлением."""
    for i in range(len(hist) - 1, 1, -1):
        d1 = abs(hist[i][0] - hist[i - 1][0])
        d2 = abs(hist[i - 1][0] - hist[i - 2][0])
        if d1 > 1e-13 and d2 > 1e-30:
            return d1 / (d2 * d2)
    return None


def task1():
    print("=" * 70)
    print("ЗАДАНИЕ 1. ОТДЕЛЕНИЕ КОРНЕЙ И УТОЧНЕНИЕ МЕТОДОМ НЬЮТОНА")
    print("f(x) = 10*cos(x) - 0.1*x^2,  отрезок [%g; %g]" % (A_LEFT, B_RIGHT))
    print("Шаг сетки отделения h = %g (фиксирован, обоснован анализом f)" % H_STEP)
    print("=" * 70)

    t0 = time.perf_counter()
    segs = isolate(A_LEFT, B_RIGHT, H_STEP)
    ngrid = int(math.ceil((B_RIGHT - A_LEFT) / H_STEP)) + 1
    print("\nОтделено корней: %d (узлов сетки: %d, вычислений f на отделении: %d)"
          % (len(segs), ngrid, ngrid))
    for i, s in enumerate(segs, 1):
        print("  отрезок %d: [%9.6f ; %9.6f]" % (i, s[0], s[1]))

    total_f = len(segs)
    total_d1 = 0
    all_res = []
    for idx, (sa, sb) in enumerate(segs, 1):
        x0, how = pick_start(sa, sb)
        root, hist = newton(x0, EPS1, EPS2)
        nf = len(hist) + 1                    # f считаем на каждом x_k + на останове
        nd = len(hist) - 1 if len(hist) > 1 else 0
        total_f += nf
        total_d1 += nd
        dx = abs(hist[-1][0] - hist[-2][0]) if len(hist) > 1 else 0.0
        err_x = 2.0 * dx                      # оценка |xi - x_n|
        af = alpha_of(hist)
        at = abs(d2f(root)) / (2.0 * abs(d1f(root)))
        all_res.append((idx, sa, sb, x0, how, root, hist, dx, err_x, af, at, nf, nd))

        print("\nКорень %d. Отрезок [%9.6f ; %9.6f], x0 = %.6f (%s)"
              % (idx, sa, sb, x0, how))
        print("  Промежуточные результаты (погрешности - в научном формате):")
        print("   k        x_k            eps_f=|f(x_k)|   eps_x=|x_k-x_(k-1)|")
        for k, (xk, fk) in enumerate(hist):
            e_f = abs(fk)
            e_x = abs(xk - hist[k - 1][0]) if k > 0 else float("nan")
            if k > 0:
                print("  %2d  %16.10f   %12.4e     %12.4e" % (k, xk, e_f, e_x))
            else:
                print("  %2d  %16.10f   %12s     %12s" % (k, xk, "-", "-"))

    el = time.perf_counter() - t0

    print("\n" + "-" * 70)
    print("Окончательные результаты (значение функции f(xi) - обычным форматом)")
    print("-" * 70)
    print("  i      xi                 f(xi)        |f(xi)|      eps_x       "
          "оценка|xi-xn|   iters   alpha_факт  alpha_теор")
    for r in all_res:
        idx, sa, sb, x0, how, root, hist, dx, err_x, af, at, nf, nd = r
        fx = func(root)
        afs = "%.6f" % af if af is not None else "  -    "
        print("  %d  %18.10f  %+12.8f  %10.2e  %10.2e  %10.2e    %2d     %s     %.6f"
              % (idx, root, fx, abs(fx), dx, err_x, len(hist) - 1, afs, at))
    print("\nВсего вычислений: f(x) = %d, f'(x) = %d; время счёта = %.4f с"
          % (total_f, total_d1, el))
    print("alpha_теор = |f''(xi)| / (2*|f'(xi)|); совпадение с alpha_факт")
    print("подтверждает квадратичную сходимость метода Ньютона.")


def gauss_solve(mat, vec):
    """Прямой ход Гаусса с выбором ведущего элемента по модулю.
    Работает с расширенной матрицей [mat | vec]. Возвращает (U, g, swaps)."""
    n = len(vec)
    aug = [list(mat[i]) + [vec[i]] for i in range(n)]
    swaps = 0
    for k in range(n):
        p = k
        for i in range(k + 1, n):
            if abs(aug[i][k]) > abs(aug[p][k]):
                p = i
        if abs(aug[p][k]) < 1e-14:
            raise ValueError("матрица вырождена")
        if p != k:
            aug[k], aug[p] = aug[p], aug[k]
            swaps += 1
        for i in range(k + 1, n):
            m = aug[i][k] / aug[k][k]
            for j in range(k, n + 1):
                aug[i][j] -= m * aug[k][j]
    return aug, swaps


def back_sub(aug):
    """Обратный ход: решение верхнетреугольной системы."""
    n = len(aug)
    sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = aug[i][n]
        for j in range(i + 1, n):
            s -= aug[i][j] * sol[j]
        sol[i] = s / aug[i][i]
    return sol


def det_gauss(mat):
    """Определитель методом Гаусса: произведение диагонали треугольной
    матрицы со знаком (-1)^s от числа перестановок строк."""
    n = len(mat)
    work = [row[:] for row in mat]
    sign = 1
    for k in range(n):
        p = k
        for i in range(k + 1, n):
            if abs(work[i][k]) > abs(work[p][k]):
                p = i
        if abs(work[p][k]) < 1e-14:
            return 0.0
        if p != k:
            work[k], work[p] = work[p], work[k]
            sign = -sign
        for i in range(k + 1, n):
            m = work[i][k] / work[k][k]
            for j in range(k, n):
                work[i][j] -= m * work[k][j]
    d = float(sign)
    for k in range(n):
        d *= work[k][k]
    return d


def inverse_gauss(mat):
    """Обратная матрица: Гаусс-Жордан на расширенной матрице [A | E]."""
    n = len(mat)
    aug = [list(mat[i]) + [1.0 if i == j else 0.0 for j in range(n)]
           for i in range(n)]
    for k in range(n):
        p = k
        for i in range(k + 1, n):
            if abs(aug[i][k]) > abs(aug[p][k]):
                p = i
        if abs(aug[p][k]) < 1e-14:
            raise ValueError("матрица вырождена, обратной не существует")
        aug[k], aug[p] = aug[p], aug[k]
        c = aug[k][k]
        for j in range(2 * n):
            aug[k][j] /= c
        for i in range(n):
            if i == k:
                continue
            c2 = aug[i][k]
            for j in range(2 * n):
                aug[i][j] -= c2 * aug[k][j]
    return [row[n:] for row in aug]


def show_mat(title, rows, w=12, p=6):
    print(title)
    for row in rows:
        print("   " + " ".join("%*.*f" % (w, p, v) for v in row))


def task2():
    print("=" * 70)
    print("ЗАДАНИЕ 2. ЛИНЕЙНАЯ АЛГЕБРА: МЕТОД ГАУССА (вариант 10)")
    print("=" * 70)

    # --- 2.1 решение СЛАУ
    print("\n2.1. Решение системы Ax = b")
    show_mat("Исходная расширенная матрица [A|b]:",
             [MAT_A[i] + [VEC_B[i]] for i in range(4)])
    aug, sw = gauss_solve(MAT_A, VEC_B)
    show_mat("После прямого хода (верхнетреугольная):", aug)
    print("Число перестановок строк: %d" % sw)
    sol = back_sub(aug)
    print("\nРешение:")
    for i, v in enumerate(sol, 1):
        print("  x%d = %.10f" % (i, v))
    res = []
    for i in range(4):
        s = sum(MAT_A[i][j] * sol[j] for j in range(4)) - VEC_B[i]
        res.append(s)
    print("Невязка r = Ax - b:")
    for i, v in enumerate(res, 1):
        print("  r%d = %+.4e" % (i, v))
    nb = max(abs(v) for v in VEC_B)
    nr = max(abs(v) for v in res)
    print("||r||_inf = %.4e, относительная невязка = %.4e" % (nr, nr / nb))

    # --- 2.2 определитель
    print("\n2.2. Определитель матрицы A методом Гаусса")
    d = det_gauss(MAT_A)
    print("det(A) = %.6f" % d)

    # --- 2.3 обратная матрица
    print("\n2.3. Обратная матрица A^(-1) (метод Гаусса-Жордана)")
    inv = inverse_gauss(MAT_A)
    show_mat("A^(-1) =", inv, w=13, p=8)
    prod = [[sum(MAT_A[i][k] * inv[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)]
    show_mat("Контроль A * A^(-1):", prod, w=13, p=8)
    dev = max(abs(prod[i][j] - (1.0 if i == j else 0.0))
              for i in range(4) for j in range(4))
    print("Невязка ||A*A^(-1) - E||_max = %.4e" % dev)


MENU = """
======================================================================
 ЛР N1, вариант 10
   1 - Задание 1 (корни уравнения, метод Ньютона)
   2 - Задание 2 (Гаусс: СЛАУ, определитель, обратная матрица)
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
        elif c == "0":
            break
        else:
            print("Нет такого пункта.")
