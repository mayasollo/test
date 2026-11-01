#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа № 1
Дисциплина: «Вычислительная математика», вариант 10.

Задание 1.   Отделение корней уравнения f(x) = 0 на отрезке [a, b] и их
             уточнение методом Ньютона.
                 f(x)  = 10*cos(x) - 0.1*x^2
                 f'(x) = -10*sin(x) - 0.2*x
                 f''(x) = -10*cos(x) - 0.2
             Входные данные: границы отрезка [a, b], шаг сетки отделения h
             (значение по умолчанию h = 0.25 обосновано анализом функции),
             точности eps1 (по аргументу) и eps2 (по функции).
             Пользователь задаёт именно шаг, а не число разбиений; величина
             шага не запрашивается как производный параметр K.

Задание 2.1. Решение системы линейных уравнений Ax = b методом Гаусса
             с выбором ведущего элемента.
Задание 2.2. Вычисление определителя матрицы A методом Гаусса.
Задание 2.3. Построение обратной матрицы A^(-1) методом Гаусса.

Запуск:  python3 lr1.py
"""

from __future__ import annotations

import math
import time

EPS_SINGULAR = 1e-12      # порог обращения в нуль ведущего элемента / f'(x)
MAX_ITER = 100            # предельное число итераций метода Ньютона

# --- данные варианта 10 -----------------------------------------------------
A_DEFAULT = -10.0
B_DEFAULT = 10.0
H_DEFAULT = 0.25          # обоснованный шаг сетки отделения корней

MATRIX_A = [[7.9, 5.6, 5.7, -7.2],
            [8.5, -4.8, 0.8, 3.5],
            [4.3, 4.2, -3.2, 9.3],
            [3.2, -1.4, -8.9, 3.3]]

VECTOR_B = [6.68, 9.95, 8.6, 1.0]


def func(x):
    """f(x) = 10*cos(x) - 0.1*x^2."""
    return 10.0 * math.cos(x) - 0.1 * x * x


def d1(x):
    """Первая производная f'(x) = -10*sin(x) - 0.2*x."""
    return -10.0 * math.sin(x) - 0.2 * x


def d2(x):
    """Вторая производная f''(x) = -10*cos(x) - 0.2."""
    return -10.0 * math.cos(x) - 0.2


def ask(prompt):
    """Запрос строки; при закрытии потока ввода (EOF) корректно завершает работу."""
    try:
        return input(prompt)
    except EOFError:
        raise SystemExit("\nВвод прерван, работа завершена.")


def ask_float(text, default):
    """Запрос вещественного числа; пустой ввод означает значение по умолчанию."""
    while True:
        raw = ask(f"{text} [{default}]: ").strip()
        if raw == "":
            return float(default)
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            print("  Нужно ввести число, попробуйте ещё раз.")


def ask_int(text, default):
    """Запрос натурального числа."""
    while True:
        raw = ask(f"{text} [{default}]: ").strip()
        if raw == "":
            return int(default)
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
        print("  Нужно ввести положительное целое число.")


def task1():
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 1. ОТДЕЛЕНИЕ КОРНЕЙ И УТОЧНЕНИЕ МЕТОДОМ НЬЮТОНА")
    print("=" * 70)
    print("Уравнение варианта 10: f(x) = 10*cos(x) - 0.1*x^2 = 0")

    a = ask_float("Левая граница отрезка a", A_DEFAULT)
    b = ask_float("Правая граница отрезка b", B_DEFAULT)
    if b <= a:
        print("Отрезок задан неверно (нужно b > a), используются значения по умолчанию.")
        a, b = A_DEFAULT, B_DEFAULT

    print("Шаг h задаётся пользователем. Обоснование выбора h:")
    print("  f'(x) = -10*sin(x) - 0.2*x, точки экстремума f(x) лежат вблизи")
    print("  нулей sin(x), т.е. в точках x ~ k*pi; наименьший полуинтервал")
    print("  монотонности между соседними экстремумами равен ~pi/2*2 = pi ~ 3.14.")
    print("  На каждом таком интервале функция имеет не более одного корня.")
    print("  Шаг h = 0.25 обеспечивает около 12 узлов между соседними")
    print("  экстремумами, надёжно фиксирует смену знака у всех корней")
    print("  и не приводит к лишнему росту вычислений; это оптимальное")
    print("  значение шага для данной функции.")
    h = ask_float("Шаг сетки отделения h", H_DEFAULT)
    if h <= 0 or h >= (b - a):
        print("Шаг задан неверно, используется h = 0.25.")
        h = H_DEFAULT

    eps1 = ask_float("Точность по аргументу eps1", 1e-8)
    eps2 = ask_float("Точность по функции eps2", 1e-8)

    n = max(1, math.ceil((b - a) / h))
    xn = [a]
    for i in range(1, n + 1):
        xn.append(b if i == n else a + i * h)

    print("\n--- Этап 1. Отделение корней ---")
    print(f"Число шагов сетки n = {n}, фактический последний шаг = "
          f"{xn[-1] - xn[-2]:.6f}")

    fn = [func(x) for x in xn]
    scan_calls = len(fn)
    segments = []
    for i in range(len(xn) - 1):
        if fn[i] == 0.0:
            segments.append((xn[i], xn[i]))
        elif fn[i] * fn[i + 1] < 0.0:
            segments.append((xn[i], xn[i + 1]))
    if fn[-1] == 0.0 and (not segments or segments[-1][1] != xn[-1]):
        segments.append((xn[-1], xn[-1]))

    if not segments:
        print("На заданном отрезке корней не обнаружено.")
        return

    print("Отрезки отделения корней и значения функции на концах:")
    print(f"{'N':>3} {'a_i':>12} {'b_i':>12} {'f(a_i)':>14} {'f(b_i)':>14}")
    for i, (l, r) in enumerate(segments, 1):
        print(f"{i:>3} {l:>12.6f} {r:>12.6f} {func(l):>14.6f} {func(r):>14.6f}")

    print("\n--- Этап 2. Уточнение корней методом Ньютона ---")
    summary = []
    total_f = total_d1 = total_d2 = 0
    started = time.perf_counter()

    for seg_index, (l, r) in enumerate(segments, 1):
        fl, fr = func(l), func(r)
        cl, cr = fl * d2(l), fr * d2(r)
        if cl > 0.0:
            x0 = l
            reason = f"условие (а): f(a)*f''(a) = {cl:.4f} > 0, x0 = a"
        elif cr > 0.0:
            x0 = r
            reason = f"условие (б): f(b)*f''(b) = {cr:.4f} > 0, x0 = b"
        else:
            x0 = l if cl >= cr else r
            reason = ("ни (а), ни (б) не выполнены; выбран конец с большим"
                      f" произведением f*f'' ({max(cl, cr):.4f})")
        calls_f = calls_d1 = calls_d2 = 2 + 2  # f и f'' на концах при выборе x0

        print(f"\nКорень {seg_index}, отрезок [{l:.6f}; {r:.6f}]")
        print(f"  Начальное приближение: x0 = {x0:.6f} ({reason})")

        history = [(x0, func(x0))]
        calls_f += 1
        x_prev = x0
        f_prev = history[0][1]
        converged = False
        attempts = 1

        for k in range(1, MAX_ITER + 1):
            g = d1(x_prev)
            calls_d1 += 1
            if abs(g) < EPS_SINGULAR:
                break
            x_new = x_prev - f_prev / g
            f_new = func(x_new)
            calls_f += 1
            delta = abs(x_new - x_prev)
            err_func = abs(f_new)
            inside = "да" if l <= x_new <= r else "нет"
            print(f"  k={k:<3} x_k={x_new:>15.10f}  "
                  f"погрешность по функции |f(x_k)|={err_func:.4e}  "
                  f"|x_k-x_(k-1)|={delta:.4e}  "
                  f"погрешность корня 2*|x_k-x_(k-1)|={2*delta:.4e}  "
                  f"x_k в отрезке: {inside}")
            history.append((x_new, f_new))
            if delta < eps1 and err_func < eps2:
                converged = True
                break
            x_prev, f_prev = x_new, f_new

        root = history[-1][0]
        f_root = history[-1][1]
        last_delta = (abs(history[-1][0] - history[-2][0])
                      if len(history) > 1 else 0.0)
        iters = len(history) - 1

        alpha_fact = None
        if len(history) >= 3:
            step_last = abs(history[-1][0] - history[-2][0])
            step_prev = abs(history[-2][0] - history[-3][0])
            if step_prev > 1e-12 and step_last > 1e-15:
                alpha_fact = step_last / step_prev ** 2
        slope = d1(root)
        alpha_theory = abs(d2(root)) / (2.0 * abs(slope)) if abs(slope) > 1e-14 else None
        calls_d2 += 1

        print(f"  Число итераций: {iters}, "
              f"сходимость: {'да' if converged else 'нет'}")
        summary.append((seg_index, l, r, root, abs(f_root), last_delta,
                        2.0 * last_delta, alpha_fact, alpha_theory,
                        calls_f, calls_d1, calls_d2))
        total_f += calls_f
        total_d1 += calls_d1
        total_d2 += calls_d2

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    print("\n--- Итоговые результаты ---")
    print(f"{'N':>3} {'xi':>18} {'f(xi)':>18} {'|f(xi)|':>12} "
          f"{'|dx|':>12} {'2*|dx|':>12} {'alpha факт':>12} {'alpha теор':>12}")
    for row in summary:
        i, l, r, xi, af, dx, est, afct, atc, *_ = row
        s_afct = "н/д" if afct is None else f"{afct:.6f}"
        s_atc = "н/д" if atc is None else f"{atc:.6f}"
        print(f"{i:>3} {xi:>18.10f} {func(xi):>18.10f} {af:>12.4e} "
              f"{dx:>12.4e} {est:>12.4e} {s_afct:>12} {s_atc:>12}")

    print(f"\nКорней найдено: {len(summary)}")
    print(f"Число вычислений f(x): {total_f + scan_calls}, "
          f"f'(x): {total_d1}, f''(x): {total_d2}")
    print(f"Время выполнения расчёта: {elapsed_ms:.2f} мс")
    print("Погрешность корня оценена по формуле |xi - x_n| <= 2*|x_n - x_(n-1)|.")


def read_system():
    """Возвращает матрицу и правую часть: вариант 10 или пользовательские."""
    use_variant = ask("Использовать систему варианта 10? (Y/n): ").strip().lower()
    if use_variant in ("", "y", "yes", "д", "да"):
        return [row[:] for row in MATRIX_A], list(VECTOR_B)
    size = ask_int("Порядок системы n", 4)
    matrix, rhs = [], []
    print(f"Вводите по {size + 1} чисел в строке: коэффициенты и правая часть.")
    for i in range(size):
        while True:
            parts = ask(f"Строка {i + 1}: ").replace(",", " ").split()
            if len(parts) == size + 1:
                try:
                    values = [float(p) for p in parts]
                    matrix.append(values[:size])
                    rhs.append(values[size])
                    break
                except ValueError:
                    pass
            print(f"  Нужно ровно {size + 1} чисел.")
    return matrix, rhs


def show_matrix(title, rows, width=None):
    print(title)
    for row in rows:
        print("   [" + " ".join(f"{v:12.6f}" for v in row) + "]")


def gauss_solve(matrix, rhs):
    """Прямой ход с выбором ведущего элемента и обратный ход.

    Возвращает (решение, число перестановок, верхнетреугольную матрицу).
    """
    size = len(rhs)
    table = [list(matrix[i]) + [rhs[i]] for i in range(size)]
    swaps = 0
    for col in range(size):
        top = col
        for row in range(col + 1, size):
            if abs(table[row][col]) > abs(table[top][col]):
                top = row
        if abs(table[top][col]) < EPS_SINGULAR:
            raise ArithmeticError("Матрица вырождена.")
        if top != col:
            table[col], table[top] = table[top], table[col]
            swaps += 1
            print(f"\nПерестановка строк {col + 1} и {top + 1}, "
                  f"ведущий элемент {table[col][col]:.6f}")
        pivot = table[col][col]
        for row in range(col + 1, size):
            m = table[row][col] / pivot
            if m == 0.0:
                continue
            for j in range(col, size + 1):
                table[row][j] -= m * table[col][j]
        show_matrix(f"\nМатрица после шага {col + 1}:", table)

    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        s = table[row][size]
        for j in range(row + 1, size):
            s -= table[row][j] * solution[j]
        solution[row] = s / table[row][row]
    return solution, swaps, table


def task2_1():
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 2.1. РЕШЕНИЕ СЛАУ МЕТОДОМ ГАУССА")
    print("=" * 70)
    matrix, rhs = read_system()
    show_matrix("Исходная расширенная матрица [A|b]:",
                [matrix[i] + [rhs[i]] for i in range(len(rhs))])
    solution, swaps, _ = gauss_solve(matrix, rhs)

    print("\nРешение системы:")
    for i, v in enumerate(solution, 1):
        print(f"  x{i} = {v:.10f}")

    residual = [sum(matrix[i][j] * solution[j] for j in range(len(solution)))
                - rhs[i] for i in range(len(matrix))]
    norm_r = max(abs(v) for v in residual)
    norm_b = max(abs(v) for v in rhs)
    print("\nНевязка r = A*x - b:")
    for i, v in enumerate(residual, 1):
        print(f"  r{i} = {v:+.6e}")
    print(f"||r||_inf = {norm_r:.6e}, относительная невязка = {norm_r / norm_b:.6e}")
    print(f"Число перестановок строк: {swaps}")


def task2_2():
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 2.2. ОПРЕДЕЛИТЕЛЬ МАТРИЦЫ МЕТОДОМ ГАУССА")
    print("=" * 70)
    matrix, _ = read_system()
    size = len(matrix)
    work = [row[:] for row in matrix]
    show_matrix("Исходная матрица A:", work)

    swaps = 0
    det = 1.0
    for col in range(size):
        top = col
        for row in range(col + 1, size):
            if abs(work[row][col]) > abs(work[top][col]):
                top = row
        if abs(work[top][col]) < EPS_SINGULAR:
            print("\nВедущий элемент равен нулю -> det(A) = 0")
            return
        if top != col:
            work[col], work[top] = work[top], work[col]
            swaps += 1
            print(f"\nПерестановка строк {col + 1} и {top + 1}")
        pivot = work[col][col]
        det *= pivot
        for row in range(col + 1, size):
            m = work[row][col] / pivot
            for j in range(col, size):
                work[row][j] -= m * work[col][j]
        show_matrix(f"\nМатрица после шага {col + 1}:", work)

    if swaps % 2:
        det = -det
    print(f"\nПроизведение ведущих элементов со знаком: det(A) = {det:.6f}")

    if size == 2:
        check = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    elif size == 3:
        check = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
                 - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
                 + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
    else:
        check = None
    if check is not None:
        print(f"Контрольное вычисление по правилу Саррюса: {check:.6f}")


def task2_3():
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 2.3. ОБРАТНАЯ МАТРИЦА МЕТОДОМ ГАУССА")
    print("=" * 70)
    matrix, _ = read_system()
    size = len(matrix)
    table = [list(matrix[i]) + [1.0 if i == j else 0.0 for j in range(size)]
             for i in range(size)]
    show_matrix("Расширенная матрица [A|E]:", table)

    for col in range(size):
        top = col
        for row in range(col + 1, size):
            if abs(table[row][col]) > abs(table[top][col]):
                top = row
        if abs(table[top][col]) < EPS_SINGULAR:
            raise ArithmeticError("Матрица вырождена, обратной не существует.")
        if top != col:
            table[col], table[top] = table[top], table[col]
            print(f"\nПерестановка строк {col + 1} и {top + 1}")
        pivot = table[col][col]
        table[col] = [v / pivot for v in table[col]]
        for row in range(size):
            if row == col:
                continue
            m = table[row][col]
            if m == 0.0:
                continue
            table[row] = [table[row][j] - m * table[col][j] for j in range(2 * size)]
        show_matrix(f"\nМатрица после шага {col + 1}:", table)

    inverse = [row[size:] for row in table]
    print("\nОбратная матрица A^(-1):")
    show_matrix("", inverse)

    product = [[sum(matrix[i][k] * inverse[k][j] for k in range(size))
                for j in range(size)] for i in range(size)]
    print("Контрольная матрица A*A^(-1):")
    show_matrix("", product)
    deviation = max(abs(product[i][j] - (1.0 if i == j else 0.0))
                    for i in range(size) for j in range(size))
    print(f"Невязка ||A*A^(-1) - E|| = {deviation:.6e}")


MENU = """
======================================================================
 ЛАБОРАТОРНАЯ РАБОТА № 1, вариант 10
======================================================================
  1 - Задание 1.   Отделение корней и метод Ньютона
  2 - Задание 2.1. Решение СЛАУ методом Гаусса
  3 - Задание 2.2. Определитель матрицы
  4 - Задание 2.3. Обратная матрица
  0 - Выход
"""

TASKS = {"1": task1, "2": task2_1, "3": task2_2, "4": task2_3}


def main():
    while True:
        print(MENU)
        choice = ask("Выберите задание: ").strip()
        if choice == "0":
            print("Работа завершена.")
            return
        task = TASKS.get(choice)
        if task is None:
            print("Нет такого пункта меню.")
            continue
        try:
            task()
        except (ValueError, ArithmeticError) as error:
            print(f"\nОшибка: {error}")
        ask("\nНажмите Enter для возврата в меню...")


if __name__ == "__main__":
    main()
