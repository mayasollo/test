#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Лабораторная работа N 2, вариант 10. Написана с нуля, только модуль math.

Задание 1 (п.6.3): интерполирование табличной функции y = sin^2(x)+1 на
[0; pi/2] (сетка h = pi/20) полиномом Ньютона m-го порядка; вычисление на
сетке x_j = j*h/2 и сравнение фактической погрешности с оценкой через
узловой многочлен omega(x).

Задание 2 (п.6.4): численное дифференцирование - первая и вторая
производные находятся аналитически из того же полинома Ньютона и
сравниваются с точными y' = sin(2x), y'' = 2*cos(2x).

Задание 3.1 (п.6.5): интеграл I = int_0^2 exp(-x^2)*cos(x) dx по составной
формуле трапеций; число отрезков удваивается до выполнения критерия
Ричардсона |I_2n - I_n|/3 <= eps*|I_2n|.

Задание 3.2 (п.6.5): тот же интеграл по формуле Гаусса n-го порядка;
узлы (корни P_n) и веса считаются самостоятельно: рекуррентный многочлен
Лежандра + итерация Ньютона, веса w_i = 2/((1-t_i^2)*P'_n(t_i)^2);
дробление отрезка уточняется удвоением с поправкой Ричардсона.

Запуск: python3 lr2.py ; пустая строка ввода - значение по умолчанию.
"""

import math

PI = math.pi
H0 = PI / 20.0                       # шаг исходной таблицы варианта 10
XA = 0.0                             # левый конец отрезка интегрирования
XB = 2.0                             # правый конец отрезка интегрирования


def zapros_chislo(nadpisi, po_umolchaniyu):
    """Вещественное число с клавиатуры; пустой ввод - default."""
    while True:
        try:
            otvet = input(f"{nadpisi} [{po_umolchaniyu}]: ").strip().replace(",", ".")
        except EOFError:
            raise SystemExit("\nВвод закрыт - завершение.")
        if not otvet:
            return float(po_umolchaniyu)
        try:
            return float(otvet)
        except ValueError:
            print("  Повторите: требуется число.")


def zapros_celoe(nadpisi, po_umolchaniyu, granicy):
    """Целое число в диапазоне [low; high]; пустой ввод - default."""
    low, high = granicy
    while True:
        try:
            otvet = input(f"{nadpisi} [{po_umolchaniyu}]: ").strip()
        except EOFError:
            raise SystemExit("\nВвод закрыт - завершение.")
        if not otvet:
            return int(po_umolchaniyu)
        try:
            znachenie = int(otvet)
        except ValueError:
            print(f"  Требуется целое от {low} до {high}.")
            continue
        if low <= znachenie <= high:
            return znachenie
        print(f"  Вне диапазона ({low}..{high}), повторите.")


def uzel_y(x):
    """Табличная функция варианта: y = sin^2(x) + 1."""
    sinus = math.sin(x)
    return sinus * sinus + 1.0


def tabliza(kolichestvo):
    """Узлы i*H0, i = 0..kolichestvo-1, и их значения."""
    uzly = [i * H0 for i in range(kolichestvo)]
    return uzly, [uzel_y(t) for t in uzly]


def raznosti(xs, ys):
    """Диагональ треугольника разделённых разностей: c[k]=f[x_0..x_k].

    Каждый следующий слой считается из предыдущего (prev -> cur),
    исходные данные не портятся.
    """
    kolichestvo = len(xs)
    c = [ys[0]]
    prev = list(ys)
    dlin = kolichestvo - 1                 # длина текущего слоя разностей
    for step in range(1, kolichestvo):
        cur = []
        for i in range(dlin):
            cur.append((prev[i + 1] - prev[i]) / (xs[i + step] - xs[i]))
        c.append(cur[0])
        prev = cur
        dlin -= 1
    return c


def blizhayshie(xs, tochka, kolichestvo):
    """kolichestvo ближайших к точке узлов, возвращаются по возрастанию."""
    pary = sorted(zip([abs(t - tochka) for t in xs], xs))[:kolichestvo]
    return [para[1] for para in sorted(pary)]


class Nyuton:
    """Полином Ньютона P(x)=sum c_k w_k(x), w_k = prod_{i<k}(x-x_i).

    При построении хранятся коэффициенты базисных многочленов w_k и их
    производных w'_k, w''_k (по степеням x, младшая степень первой).
    Переход w_{k+1} = w_k*(x - x_k) даёт рекуррентно
    w'_{k+1} = w'_k*(x-x_k) + w_k и w''_{k+1} = w''_k*(x-x_k) + 2*w'_k.
    Тогда P' = sum c_k w'_k, P'' = sum c_k w''_k.
    """

    def __init__(self, xs, c):
        self.xs = xs
        self.c = c
        n = len(c)
        w, wp, wpp = [1.0], [0.0], [0.0]
        rows_w, rows_wp, rows_wpp = [], [], []
        for k in range(n):
            rows_w.append(list(w))
            rows_wp.append(list(wp))
            rows_wpp.append(list(wpp))
            if k == n - 1:
                break
            s = -xs[k]
            nw = [0.0] * (k + 2)
            nwp = [0.0] * (k + 2)
            nwpp = [0.0] * (k + 2)
            for d in range(k + 1):
                nw[d + 1] += w[d]
                nw[d] += s * w[d]
                nwp[d + 1] += wp[d]
                nwp[d] += s * wp[d] + w[d]
                nwpp[d + 1] += wpp[d]
                nwpp[d] += s * wpp[d] + 2.0 * wp[d]
            w, wp, wpp = nw, nwp, nwpp
        self.rows_w, self.rows_wp, self.rows_wpp = rows_w, rows_wp, rows_wpp

    @staticmethod
    def _gon(skoef, x):
        """Схема Горнера для коэффициентов (младшая степень первой)."""
        itog = 0.0
        for a in reversed(skoef):
            itog = itog * x + a
        return itog

    def znachenie(self, x):
        return sum(self.c[k] * self._gon(self.rows_w[k], x)
                   for k in range(len(self.c)))

    def proizvodnaya1(self, x):
        return sum(self.c[k] * self._gon(self.rows_wp[k], x)
                   for k in range(len(self.c)))

    def proizvodnaya2(self, x):
        return sum(self.c[k] * self._gon(self.rows_wpp[k], x)
                   for k in range(len(self.c)))

    def omega(self, x):
        """Узловой многочлен prod (x - x_i)."""
        result = 1.0
        for t in self.xs:
            result *= (x - t)
        return result


def zadanie1():
    print("\n" + "=" * 72)
    print("ЗАДАНИЕ 1. ИНТЕРПОЛИРОВАНИЕ ПОЛИНОМОМ НЬЮТОНА")
    print("=" * 72)
    print("y = sin^2(x)+1, сетка h=pi/20 (11 узлов), расчёт на x_j = j*h/2")
    m = zapros_celoe("Порядок полинома m", 3, (1, 9))

    xs_all, ys_all = tabliza(11)
    print("\nИсходная таблица:")
    for i in range(11):
        print(f"  x_{i:<2}= {xs_all[i]:9.6f}   y_{i}= {ys_all[i]:11.8f}")

    factorial = 1
    for q in range(2, m + 2):
        factorial *= q                    # (m+1)!
    M = 2.0 ** m                          # max |y^(m+1)| на [0; pi/2]

    print(f"\nПостроение P_m по {m + 1} ближайшим узлам, m = {m}:")
    print("  j     x_j      y(x_j)     P_m(x_j)    |y-P_m|      ocenka")
    fakty_max = 0.0
    ocenka_max = 0.0
    for j in range(21):
        xj = j * H0 / 2.0
        xs = blizhayshie(xs_all, xj, m + 1)
        poly = Nyuton(xs, raznosti(xs, [uzel_y(t) for t in xs]))
        p = poly.znachenie(xj)
        otkl = abs(uzel_y(xj) - p)
        ocenka = M / factorial * abs(poly.omega(xj))
        fakty_max = max(fakty_max, otkl)
        ocenka_max = max(ocenka_max, ocenka)
        print(f"  {j:>2}  {xj:9.6f}  {uzel_y(xj):10.8f}  {p:10.8f} "
              f"{otkl:10.3e}  {ocenka:10.3e}")

    verdict = ("выполняется" if fakty_max <= ocenka_max * 1.0000001
               else "НАРУШАЕТСЯ")
    print(f"\nМаксимум |y-P_m| = {fakty_max:.6e}, максимум оценки = {ocenka_max:.6e}")
    print(f"Неравенство |y-P_m| <= M_(m+1)/(m+1)!*|omega(x)| {verdict}.")


def zadanie2():
    print("\n" + "=" * 72)
    print("ЗАДАНИЕ 2. ЧИСЛЕННОЕ ДИФФЕРЕНЦИРОВАНИЕ")
    print("=" * 72)
    m = zapros_celoe("Порядок полинома m", 3, (1, 9))
    xs_all, _ = tabliza(11)

    print(f"P'_m против y'=sin(2x) и P''_m против y''=2cos(2x), m = {m}:")
    print("  j     x_j      y'_true     P'_m       otl1      y''_true"
          "     P''_m      otl2")
    e1 = 0.0
    e2 = 0.0
    for j in range(21):
        xj = j * H0 / 2.0
        xs = blizhayshie(xs_all, xj, m + 1)
        poly = Nyuton(xs, raznosti(xs, [uzel_y(t) for t in xs]))
        d1 = poly.proizvodnaya1(xj)
        d2 = poly.proizvodnaya2(xj)
        tochno1 = math.sin(2.0 * xj)
        tochno2 = 2.0 * math.cos(2.0 * xj)
        o1 = abs(tochno1 - d1)
        o2 = abs(tochno2 - d2)
        e1 = max(e1, o1)
        e2 = max(e2, o2)
        print(f"  {j:>2}  {xj:9.6f}  {tochno1:10.7f}  {d1:10.7f} {o1:9.2e} "
              f" {tochno2:10.7f}  {d2:10.7f} {o2:9.2e}")

    print(f"\nmax |y'-P'_m| = {e1:.6e};  max |y''-P''_m| = {e2:.6e}")


def pod_int(x):
    """Подынтегральная функция варианта: exp(-x^2)*cos(x)."""
    return math.exp(-(x * x)) * math.cos(x)


def trapeziya(chislo_otrezkov):
    """Составная трапеция на [XA; XB], chislo_otrezkov равных частей."""
    shag = (XB - XA) / chislo_otrezkov
    summ = 0.5 * (pod_int(XA) + pod_int(XB))
    k = 1
    while k < chislo_otrezkov:
        summ += pod_int(XA + k * shag)
        k += 1
    return summ * shag


def zadanie3_1():
    print("\n" + "=" * 72)
    print("ЗАДАНИЕ 3.1. ФОРМУЛА ТРАПЕЦИЙ С АВТОМАТИЧЕСКИМ ШАГОМ")
    print("=" * 72)
    print("I = int_0^2 exp(-x^2)*cos(x) dx; критерий |I_2n-I_n|/3 <= eps*|I_2n|")
    eps = zapros_chislo("eps (относительная точность)", 1e-6)

    chislo = 2
    pred = trapeziya(chislo)
    print(f"\n     n            I_n       ocena_R      otnosh")
    print(f"  {chislo:>5}  {pred:14.10f}          -          -")
    while True:
        chislo *= 2
        tekush = trapeziya(chislo)
        ocena = abs(tekush - pred) / 3.0
        otnosh = ocena / abs(tekush)
        print(f"  {chislo:>5}  {tekush:14.10f}  {ocena:10.3e}  {otnosh:10.3e}")
        pred = tekush
        if otnosh <= eps or chislo >= 2 ** 22:
            break
    print(f"\nОтвет: I = {tekush:.10f}, n = {chislo}, "
          f"h = {(XB - XA) / chislo:.3e}, otnosh = {otnosh:.3e} <= eps = {eps:.3e}")


def lezhandr(n, t):
    """P_n(t) по трёхточечной рекурренте (k+1)P_{k+1}=(2k+1)tP_k-kP_{k-1}."""
    a, b = 1.0, t
    if n == 0:
        return a
    for k in range(1, n):
        a, b = b, ((2 * k + 1) * t * b - k * a) / (k + 1)
    return b


def pro_lezhandra(n, t):
    """P'_n(t) = n*(t*P_n(t) - P_{n-1}(t)) / (t^2 - 1)."""
    znam = t * t - 1.0
    if abs(znam) < 1e-15:
        return 0.5 * n * (n + 1) * (1.0 if t > 0 else (-1.0) ** (n + 1))
    return n * (t * lezhandr(n, t) - lezhandr(n - 1, t)) / znam


def tablica_gaussa(n):
    """Узлы - корни P_n (старт cos(pi*(i+0.75)/(n+0.5)), итерации Ньютона),
    веса - w_i = 2 / ((1 - t_i^2) * P'_n(t_i)^2)."""
    uzly = []
    vesa = []
    for i in range(n):
        t = math.cos(PI * (i + 0.75) / (n + 0.5))
        for _ in range(50):
            prirashchenie = lezhandr(n, t) / pro_lezhandra(n, t)
            t -= prirashchenie
            if abs(prirashchenie) < 1e-17:
                break
        uzly.append(t)
        vesa.append(2.0 / ((1.0 - t * t) * pro_lezhandra(n, t) ** 2))
    return uzly, vesa


def gauss_na_otrezke(uzly, vesa, lev, prav):
    """Одно применение формулы Гаусса на [lev; prav] (аффинное отображение)."""
    polovina = 0.5 * (prav - lev)
    seredina = 0.5 * (prav + lev)
    summ = 0.0
    for idx in range(len(uzly)):
        summ += vesa[idx] * pod_int(seredina + polovina * uzly[idx])
    return summ * polovina


def zadanie3_2():
    print("\n" + "=" * 72)
    print("ЗАДАНИЕ 3.2. КВАДРАТУРА ГАУССА")
    print("=" * 72)
    n = zapros_celoe("Порядок формулы Гаусса n", 6, (1, 12))
    eps = zapros_chislo("eps (относительная точность)", 1e-10)

    uzly, vesa = tablica_gaussa(n)
    print(f"\nУзлы/веса {n}-й гауссианы на [-1;1], контроль P_n(t_i) ~ 0:")
    for i in range(n):
        print(f"  t_{i} = {uzly[i]:+16.14f}   w_{i} = {vesa[i]:+16.14f}"
              f"   P_n = {lezhandr(n, uzly[i]):+.2e}")
    print(f"Контроль суммы весов: sum w_i = {sum(vesa):.14f} (точно 2)")

    delitel = 4.0 ** n - 1.0              # 2^(2n)-1 - знаменатель Ричардсона
    chasti = 1
    pred = gauss_na_otrezke(uzly, vesa, XA, XB)
    print(f"\n     N            I_N         popravka     otnosh")
    print(f"  {chasti:>5}  {pred:16.14f}          -          -")
    while True:
        chasti *= 2
        tekush = 0.0
        shag = (XB - XA) / chasti
        for k in range(chasti):
            tekush += gauss_na_otrezke(uzly, vesa,
                                       XA + k * shag, XA + (k + 1) * shag)
        ocena = abs(tekush - pred) / delitel
        otnosh = ocena / abs(tekush)
        print(f"  {chasti:>5}  {tekush:16.14f}  {ocena:11.3e}  {otnosh:11.3e}")
        pred = tekush
        if otnosh <= eps or chasti >= 2 ** 18:
            break
    print(f"\nОтвет: I = {tekush:.14f}, n = {n}, N = {chasti}, "
          f"otnosh = {otnosh:.3e} <= eps = {eps:.3e}")


SPISOK = [(1, "Интерполяция полиномом Ньютона (зад. 1)", zadanie1),
          (2, "Численное дифференцирование (зад. 2)", zadanie2),
          (3, "Формула трапеций, автошаг (зад. 3.1)", zadanie3_1),
          (4, "Формула Гаусса (зад. 3.2)", zadanie3_2),
          (0, "Выход", None)]


def menyu():
    while True:
        print("\n" + "=" * 72)
        print(" ЛР N 2, вариант 10 - выберите задание")
        print("=" * 72)
        for nomer, opisanie, _ in SPISOK:
            print(f"   {nomer} - {opisanie}")
        vybor = zapros_chislo("Пункт", 0)
        found = False
        for nomer, _, funkcija in SPISOK:
            if int(vybor) == nomer:
                found = True
                if funkcija is None:
                    print("Работа завершена.")
                    return
                funkcija()
                break
        if not found:
            print("Нет такого пункта.")


if __name__ == "__main__":
    menyu()
