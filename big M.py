import numpy as np
import pandas as pd

INFINITY = 10e9


def take_input():
    # print("Enter the number of variables in the objective function")
    n = int(input())
    # print("Enter the number of constraints")
    m = int(input())
    # print("Enter the coefficients of the Objective Function")
    c = [float(i) for i in input().split(" ")]
    # print("Enter the value of the constant in the objective function")
    d = float(input())
    # print("Enter the matrix A which is the coefficient of the constraints row by row")
    a = []
    for i in range(m):
        a.append([float(j) for j in input().split(" ")])
    # print("Enter the constants/RHS bi for the constraint equations")
    b = [float(i) for i in input().split(" ")]
    # print("Enter the equation or in-equation type fo the variable")
    s = input().split(" ")
    return n, m, a, c, d, b, s


def val_objective_function(c, x, d):
    return np.dot(c, x) + d


def print_standard_form(n, m, a, c, d, b, s, M):
    obj_func = ""
    for i, c_prime in enumerate(c):
        obj_func += str(c_prime) + "x" + str(i + 1)
        obj_func += " + "
    size = len(c)
    obj_func += str(d)
    print("The objective function to maximize is: \n" + obj_func)
    print("\nThe standard form of the constraints using slack/surplus variables is:")
    k = n + 1
    x = []
    add_col = []
    c_hash = [l for l in c]
    list_of_basic_variables = []
    list_of_non_basic_variables = []
    for i in range(n):
        list_of_basic_variables.append("x" + str(i + 1))
        x.append(0)
    for j, row in enumerate(a):
        constraint = ""
        for i, a_prime in enumerate(row):
            constraint += str(a_prime) + "x" + str(i + 1) + " + " * (i != len(row) - 1)
        if s[j] == "=" or s[j] == "<=":
            constraint += " + x" + str(k)
            list_of_non_basic_variables.append("x" + str(k))
            x.append(b[j])
            c_hash.append(0)
            k += 1
        else:
            constraint += " - x" + str(k) + " + x" + str(k + 1)
            list_of_non_basic_variables.append("x" + str(k + 1))
            x.append(0)
            c_hash.append(0)
            list_of_basic_variables.append("x" + str(k))
            x.append(b[j])
            temp_col = np.zeros((m,))
            temp_col[j] = -1
            add_col.append(temp_col)
            c.append(-M)
            c_hash.append((-M))
            k += 2
        constraint += " = " + str(b[j])
        print(constraint)
    for alpha in range(m):
        for beta in range(n):
            c[beta] -= a[alpha][beta]*(-M)
    if len(list_of_basic_variables) - size > 0:
        additional = np.array(add_col).transpose()
        a = np.concatenate([a, additional], axis=1)
    return n, m, np.array(a), np.array(c), np.array(c_hash), d, np.array(
        b), s, list_of_basic_variables, list_of_non_basic_variables, np.array(x)


def print_table(a_prime, c_prime, d_prime, b_prime, c, list_of_basic_variables, list_of_non_basic_variables, x):
    l_ = []
    for basic in list_of_basic_variables:
        l_.append(basic + '=' + str(x[int(basic[1]) - 1]))
    print("B = ", l_)
    l_ = []
    for non_basic in list_of_non_basic_variables:
        l_.append(non_basic + '=' + str(x[int(non_basic[1]) - 1]))
    print("N = ", l_)
    print(a_prime)
    print(b_prime)
    print(-c_prime)
    print("So the optimal value of objective function is:", round(val_objective_function(c, x, d_prime), 5))


def print_pivot(cv, v, u, ratios, min_ratio, pivot):
    print("The value of c is", -cv, " corresponding to column", v + 1)
    print("The ratios are for corresponding column", ratios)
    print("The minimum ratio is:", min_ratio)
    print("The pivot element is ", pivot, " and corresponding coordinates(1 based indexing) is", u + 1, " ", v + 1)


def simplex(n_prime, m_prime, a_prime, c_prime, c, d_prime, b_prime, list_of_basic_variables, list_of_non_basic_variables,
            x):
    ite = 0
    n_prime = a.shape[1]
    m_prime = a.shape[0]
    while True:
        ite += 1
        print("-" * 60)
        print("Iteration ", ite)
        print_table(a_prime, c_prime, d_prime, b_prime, c, list_of_basic_variables, list_of_non_basic_variables, x)
        v = np.argmax(c_prime)
        cv = c_prime[v]
        if cv < 0:
            print("-" * 60)
            print("The iterations have ended")
            print("This is the list of all the basic variables are ", list_of_basic_variables)
            print("This is the list of all non-basic variables are ", list_of_non_basic_variables)
            print("The values for x are:")
            li = []
            for i in range(len(x)):
                li.append("x" + str(i + 1) + '=' + str(round(x[i], 5)))
            print(li)
            print("So the Final value of objective function is:", round(val_objective_function(c, x, d_prime), 5))
            return
        if cv == 0:
            x_1 = np.copy(x)
            ratios = np.empty((m_prime,))
            u = 0
            pivot = -1
            min_ratio = INFINITY
            for i in range(m_prime):
                if a_prime[i][v] == 0:
                    continue
                ratios[i] = b_prime[i] / a_prime[i][v]
                if min_ratio > ratios[i] > 0:
                    u = i
                    min_ratio = ratios[i]
                    pivot = a_prime[u][v]
            if min_ratio == INFINITY:
                print("-" * 60)
                print("There are no more optimal solutions")
                return
            a_new = np.empty((m_prime, n_prime))
            for i in range(m_prime):
                for j in range(n_prime):
                    if i == u and j == v:
                        a_new[i][j] = 1 / a_prime[i][j]
                    elif i == u:
                        a_new[i][j] = a_prime[i][j] / pivot
                    elif j == v:
                        a_new[i][j] = -a_prime[i][j] / pivot
                    else:
                        a_new[i][j] = (pivot * a_prime[i][j] - a_prime[i][v] * a_prime[u][j]) / pivot
            c_new = np.copy(c_prime)
            for j in range(n_prime):
                if j == v:
                    c_new[j] = round(-c_prime[j] / pivot, 6)
                else:
                    c_new[j] = round((pivot * c_prime[j] - c_prime[v] * a_prime[u][j]) / pivot, 6)
            b_new = np.copy(b_prime)
            for j in range(m_prime):
                if j == u:
                    b_new[j] = b_prime[j] / pivot
                else:
                    b_new[j] = (pivot * b_prime[j] - a_prime[j][v] * b_prime[u]) / pivot
            a_prime = np.copy(a_new)
            c_prime = np.copy(c_new)
            b_prime = np.copy(b_new)
            temp1 = list_of_basic_variables[v]
            list_of_basic_variables[v] = list_of_non_basic_variables[u]
            list_of_non_basic_variables[u] = temp1
            for i in range(m_prime):
                s = list_of_non_basic_variables[i]
                x[int(s[1]) - 1] = b_prime[i]
            for i in range(n_prime):
                s = list_of_basic_variables[i]
                x[int(s[1]) - 1] = 0
            x_2 = np.copy(x)
            print_table(a_prime, c_prime, d_prime, b_prime, c, list_of_basic_variables, list_of_non_basic_variables, x)
            print("There are infinitely many solutions of the form: \u03BB", x_1, " + (1-\u03BB)", x_2)
            return
        else:
            print("The value of most negative c_prime is", -cv, " Corresponding to column", v + 1)
            ratios = np.empty((m_prime,))
            u = 0
            pivot = -1
            min_ratio = INFINITY
            for i in range(m_prime):
                if a_prime[i][v] == 0:
                    continue
                ratios[i] = b_prime[i] / a_prime[i][v]
                if min_ratio > ratios[i] >= 0:
                    u = i
                    min_ratio = ratios[i]
                    pivot = a_prime[u][v]
            print("The ratios are for corresponding column", ratios)
            print("The minimum ratio is:", min_ratio)
            print("The pivot element is ", pivot, " and corresponding coordinates(1 based indexing) is", u + 1, " ",
                  v + 1)
            if min_ratio == INFINITY:
                print("-" * 60)
                print("The problem is unbounded")
                print("The value of objective function is:", val_objective_function(c, x, d_prime))
                print("The values for x are:", x)
                return
            a_new = np.empty((m_prime, n_prime))
            for i in range(m_prime):
                for j in range(n_prime):
                    if i == u and j == v:
                        a_new[i][j] = 1 / a_prime[i][j]
                    elif i == u:
                        a_new[i][j] = a_prime[i][j] / pivot
                    elif j == v:
                        a_new[i][j] = -a_prime[i][j] / pivot
                    else:
                        a_new[i][j] = (pivot * a_prime[i][j] - a_prime[i][v] * a_prime[u][j]) / pivot
            c_new = np.copy(c_prime)
            for j in range(n_prime):
                if j == v:
                    c_new[j] = round(-c_prime[j] / pivot,6)
                else:
                    c_new[j] = round((pivot * c_prime[j] - c_prime[v] * a_prime[u][j]) / pivot,6)
            b_new = np.copy(b_prime)
            for j in range(m_prime):
                if j == u:
                    b_new[j] = b_prime[j] / pivot
                else:
                    b_new[j] = (pivot * b_prime[j] - a_prime[j][v] * b_prime[u]) / pivot
            a_prime = np.copy(a_new)
            c_prime = np.copy(c_new)
            b_prime = np.copy(b_new)
            temp1 = list_of_basic_variables[v]
            list_of_basic_variables[v] = list_of_non_basic_variables[u]
            list_of_non_basic_variables[u] = temp1
            for i in range(m_prime):
                s = list_of_non_basic_variables[i]
                x[int(s[1]) - 1] = b_prime[i]
            for i in range(n_prime):
                s = list_of_basic_variables[i]
                x[int(s[1]) - 1] = 0



if __name__ == '__main__':
    # t = int(input("Enter the number of test cases:"))
    t = 1
    M = 10000
    for i in range(t):
        print("*" * 60)
        print("This is the solution to the testcase number ", i + 1)
        n, m, a, c, d, b, s = take_input()
        # Question 1
        n, m, a, c, c_hash, d, b, s, list_of_basic_variables, list_of_non_basic_variables, x = print_standard_form(n, m, a, c,
                                                                                                           d, b, s,
                                                                                                           M)
        # simplex method
        simplex(n, m, a, c, c_hash, d, b, list_of_basic_variables, list_of_non_basic_variables, x)
