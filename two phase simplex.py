import numpy as np
import pandas as pd

INFINITY = 10e9


def take_input():
    optimization = str(input())
    # print("Enter the number of variables in the objective function")
    n = int(input())
    # print("Enter the number of constraints")
    m = int(input())
    # print("Enter the coefficients of the Objective Function")
    c = [float(i) for i in input().split(" ")]
    if optimization == 'min':
        c = [-z for z in c]
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
    return n, m, a, c, d, b, s, optimization


def val_objective_function(c, b, d, optimization):
    if optimization == 'max':
        return np.dot(np.transpose(c), b) + d
    else:
        return -np.dot(np.transpose(c), b) + d


def print_standard_form_phase_one(n, m, a, d, b, s):
    num_artificial_variables = 0
    num_surplus_variables = 0
    num_slack_variables = 0
    k = n + 1
    x = []
    add_col = []
    list_of_nonbasic_variables = []
    list_of_basic_variables = []
    list_of_slack_variables = []
    list_of_artificial_variables = []
    list_of_surplus_variables = []
    a = np.array(a)
    b = np.array(b)
    for i in range(n):
        list_of_nonbasic_variables.append("x" + str(i + 1))
        x.append(0)
    constraints = []
    for j, row in enumerate(a):
        constraint = ""
        for i, z in enumerate(row):
            constraint += str(z) + "x" + str(i + 1) + " + " * (i != len(row) - 1)
        if s[j] == "=":
            constraint += " + x" + str(k)
            list_of_basic_variables.append("x" + str(k))
            list_of_artificial_variables.append("x" + str(k))
            x.append(b[j])
            k += 1
            num_artificial_variables += 1
        elif s[j] == "<=":
            constraint += " + x" + str(k)
            list_of_basic_variables.append("x" + str(k))
            list_of_slack_variables.append("x" + str(k))
            x.append(b[j])
            k += 1
            num_slack_variables += 1
        else:
            constraint += " - x" + str(k) + " + x" + str(k + 1)
            list_of_basic_variables.append("x" + str(k + 1))
            list_of_artificial_variables.append("x" + str(k + 1))
            x.append(0)
            list_of_nonbasic_variables.append("x" + str(k))
            list_of_surplus_variables.append("x" + str(k))
            x.append(b[j])
            num_surplus_variables += 1
            temp_col = np.zeros((m,))
            temp_col[j] = -1
            add_col.append(temp_col)
            num_artificial_variables += 1
            k += 2
        constraint += " = " + str(b[j])
        constraints.append(constraint)
    if len(list_of_surplus_variables) > 0:
        additional = np.array(add_col).transpose()
        a = np.concatenate([a, additional], axis=1)

    cn = np.zeros((len(list_of_nonbasic_variables)))
    cb = np.zeros(((len(list_of_basic_variables))))
    for i, var in enumerate(list_of_basic_variables):
        if var in list_of_artificial_variables:
            cb[i] = -1
    c = np.zeros((len(list_of_nonbasic_variables)))
    for i in range(len(list_of_nonbasic_variables)):
        c[i] = -1 * (np.dot(np.transpose(cb), a[:, i]) - cn[i])
    print("%" * 90)
    print("This is Phase-1")
    print("The artificial objective function to maximize is")
    obj_func = ""
    for i, var in enumerate(list_of_artificial_variables):
        obj_func += "-" + str(var) + " "
    print(obj_func + " + ", d)
    for constraint in constraints:
        print(constraint)
    return a, b, cb, cn, c, np.array(
        x), list_of_nonbasic_variables, list_of_basic_variables, list_of_artificial_variables, list_of_slack_variables, list_of_surplus_variables


def print_table(a, c, d, b, cb, x, list_of_nonbasic_variables, list_of_basic_variables, optimization):
    l_ = []
    for basic in list_of_nonbasic_variables:
        l_.append(basic + '=' + str(x[int(basic[1]) - 1]))
    print("Non Basic Variables = ", l_)
    l_ = []
    for non_basic in list_of_basic_variables:
        l_.append(non_basic + '=' + str(x[int(non_basic[1]) - 1]))
    print("Basic Variables = ", l_)
    print("This is A martix")
    print(a)
    print("This is Xb")
    print(b)
    print("This is c bottom row of table")
    print(-c)
    print("This is Cb (coefficients of basic variable")
    print(cb)
    print("This is Cn (coefficients of non-basic variable")
    print(cn)
    print("So the optimal value of objective function is:", round(val_objective_function(cb, b, d, optimization), 5))


def print_pivot(cv, v, u, ratios, min_ratio, pivot):
    print("The value of c is", -cv, " corresponding to column", v + 1)
    print("The ratios are for corresponding column", ratios)
    print("The minimum ratio is:", min_ratio)
    print("The pivot element is ", pivot, " and corresponding coordinates(1 based indexing) is", u + 1, " ", v + 1)


def artificial_obj_function_val(artificial_variables_list, x, d, optimization):
    c = np.zeros((len(x),))
    for var in artificial_variables_list:
        c[int(var[1]) - 1] = 1
    if optimization == 'max':
        return np.dot(c, x) + d
    else:
        return -np.dot(c, x) + d


def phase_one_simplex(a, b, cb, cn, c, d, x, list_of_nonbasic_variables, list_of_basic_variables,
                      list_of_artificial_variables, list_of_slack_variables, list_of_surplus_variables):
    ite = 0
    n_prime = a.shape[1]
    m_prime = a.shape[0]
    num_artificial_variables = len(list_of_artificial_variables)
    while True:
        ite += 1
        print("-" * 60)
        print("Iteration ", ite)
        print_table(a, c, d, b, cb, x, list_of_nonbasic_variables, list_of_basic_variables, optimization)
        v = np.argmax(c)
        cv = c[v]
        if cv <= 0:
            if ite > num_artificial_variables:
                print("-" * 60)
                print("The phase one has ended")
                print("This is the list of all the basic variables are ", list_of_nonbasic_variables)
                print("This is the list of all non-basic variables are ", list_of_basic_variables)
                print("The values for x are:")
                li = []
                for i in range(len(x)):
                    li.append("x" + str(i + 1) + '=' + str(round(x[i], 5)))
                print(li)
                print("So the Final value of the artificial objective function is after phase one is:",
                      round(artificial_obj_function_val(list_of_artificial_variables, x, d, optimization), 5))
                return a, b, c, x, ite, 1
            else:
                print("-" * 60)
                print("The Solution is infeasible because all artificial variables are not zero")
                print("This is the list of all the basic variables are ", list_of_nonbasic_variables)
                print("This is the list of all non-basic variables are ", list_of_basic_variables)
                print("The values for x are:")
                li = []
                for i in range(len(x)):
                    li.append("x" + str(i + 1) + '=' + str(round(x[i], 5)))
                print(li)
                print("As you can see the the value of artificial objective function is :",
                      round(artificial_obj_function_val(list_of_artificial_variables, x, d, optimization), 5), "which "
                                                                                                               "is "
                                                                                                               "not "
                                                                                                               "zero")
                return a, b, c, x, ite, 0
        else:
            print("The value of most negative c is", -cv, " Corresponding to column", v + 1)
            ratios = np.empty((m_prime,))
            u = 0
            pivot = -1
            min_ratio = INFINITY
            for i in range(m_prime):
                if a[i][v] == 0:
                    continue
                ratios[i] = b[i] / a[i][v]
                if min_ratio > ratios[i] >= 0:
                    u = i
                    min_ratio = ratios[i]
                    pivot = a[u][v]
            print("The ratios are for corresponding column", ratios)
            print("The minimum ratio is:", min_ratio)
            print("The pivot element is ", pivot, " and corresponding coordinates(1 based indexing) is", u + 1, " ",
                  v + 1)
            if min_ratio == INFINITY:
                print("-" * 60)
                print("The problem is unbounded")
                print(print("As you can see the the value of artificial objective function is :",
                      round(artificial_obj_function_val(list_of_artificial_variables, x, d, optimization), 5)))
                print("The values for x are:", x)
                return a, b, c, x, ite, 0
            a_new = np.empty((m_prime, n_prime))
            for i in range(m_prime):
                for j in range(n_prime):
                    if i == u and j == v:
                        a_new[i][j] = 1 / a[i][j]
                    elif i == u:
                        a_new[i][j] = a[i][j] / pivot
                    elif j == v:
                        a_new[i][j] = -a[i][j] / pivot
                    else:
                        a_new[i][j] = (pivot * a[i][j] - a[i][v] * a[u][j]) / pivot
            c_new = np.copy(c)
            for j in range(n_prime):
                if j == v:
                    c_new[j] = round(-c[j] / pivot, 6)
                else:
                    c_new[j] = round((pivot * c[j] - c[v] * a[u][j]) / pivot, 6)
                if abs(c_new[j]) <= 0.0001:
                    c_new[j] = 0
            b_new = np.copy(b)
            for j in range(m_prime):
                if j == u:
                    b_new[j] = b[j] / pivot
                else:
                    b_new[j] = (pivot * b[j] - a[j][v] * b[u]) / pivot
            a = np.copy(a_new)
            c = np.copy(c_new)
            b = np.copy(b_new)
            temp1 = list_of_nonbasic_variables[v]
            list_of_nonbasic_variables[v] = list_of_basic_variables[u]
            list_of_basic_variables[u] = temp1
            temp2 = cb[u]
            cb[u] = cn[v]
            cn[v] = temp2
            for i in range(m_prime):
                s = list_of_basic_variables[i]
                x[int(s[1]) - 1] = b[i]
            for i in range(n_prime):
                s = list_of_nonbasic_variables[i]
                x[int(s[1]) - 1] = 0


def print_standard_form_phase_two(a, b, c, cb, cn, c_original, x, list_of_basic_variables, list_of_nonbasic_variables,
                                  list_of_artificial_variables, list_of_slack_variables, list_of_surplus_variables):
    print("%" * 60)
    print("This is phase two")
    list_of_removable_indices = []
    while len(c_original) < len(list_of_basic_variables) + len(list_of_nonbasic_variables):
        c_original.append(0)
    for i, var in enumerate(list_of_nonbasic_variables):
        cn[i] = c_original[int(var[1]) - 1]
    for i, var in enumerate(list_of_basic_variables):
        cb[i] = c_original[int(var[1]) - 1]
    for i in range(len(list_of_nonbasic_variables)):
        c[i] = -1 * (np.dot(np.transpose(cb), a[:, i]) - cn[i])
    for i, var in enumerate(list_of_nonbasic_variables):
        if var in list_of_artificial_variables:
            list_of_removable_indices.append(i)
    for var in list_of_artificial_variables:
        list_of_nonbasic_variables.remove(var)
    cn = np.delete(cn, list_of_removable_indices, axis=0)
    a = np.delete(a, list_of_removable_indices, axis=1)
    c = np.delete(c, list_of_removable_indices, axis=0)
    """
    print(a)
    print(b)
    print(c)
    print(cn)
    print(cb)
    print(list_of_basic_variables)
    print(list_of_nonbasic_variables)
    print(list_of_surplus_variables)
    print(list_of_slack_variables)
    print(x)
    """
    return a, b, c, cn, cb, list_of_basic_variables, list_of_nonbasic_variables, list_of_surplus_variables, list_of_slack_variables, list_of_artificial_variables


def phase_two_simplex(a, c, cn, cb, d, b, list_of_nonbasic_variables,
                      list_of_basic_variables,
                      x, optimization, num_surplus_variables, num_slack_variables, num_artificial_variables, ite):
    n_prime = a.shape[1]
    m_prime = a.shape[0]
    while True:
        ite += 1
        print("-" * 60)
        print("Iteration ", ite)
        print_table(a, c, d, b, cb, x, list_of_nonbasic_variables, list_of_basic_variables,
                    optimization)
        v = np.argmax(c)
        cv = c[v]
        if cv < 0:
            if ite > num_artificial_variables:
                print("-" * 60)
                print("The iterations have ended")
                print("This is the list of all the non-basic variables are ", list_of_nonbasic_variables)
                print("This is the list of all basic variables are ", list_of_basic_variables)
                print("The values for x are:")
                li = []
                for i in range(len(x)):
                    li.append("x" + str(i + 1) + '=' + str(round(x[i], 5)))
                print(li)
                print("So the Final value of objective function is:",
                      round(val_objective_function(cb, b, d, optimization), 5))
                return
            else:
                print("-" * 60)
                print("The Solution is infeasible because all artificial variables are not zero")
                print("This is the list of all the non-basic variables are ", list_of_nonbasic_variables)
                print("This is the list of all basic variables are ", list_of_basic_variables)
                print("The values for x are:")
                li = []
                for i in range(len(x)):
                    li.append("x" + str(i + 1) + '=' + str(round(x[i], 5)))
                print(li)
                print("As you can see the the value of objective function is :",
                      round(val_objective_function(cb, b, d, optimization), 5))
                return
        if cv == 0:
            x_1 = np.copy(x)
            ratios = np.empty((m_prime,))
            u = 0
            pivot = -1
            min_ratio = INFINITY
            for i in range(m_prime):
                if a[i][v] == 0:
                    continue
                ratios[i] = b[i] / a[i][v]
                if min_ratio > ratios[i] > 0:
                    u = i
                    min_ratio = ratios[i]
                    pivot = a[u][v]
            if min_ratio == INFINITY:
                print("-" * 60)
                print("There are no more optimal solutions")
                return
            a_new = np.empty((m_prime, n_prime))
            for i in range(m_prime):
                for j in range(n_prime):
                    if i == u and j == v:
                        a_new[i][j] = 1 / a[i][j]
                    elif i == u:
                        a_new[i][j] = a[i][j] / pivot
                    elif j == v:
                        a_new[i][j] = -a[i][j] / pivot
                    else:
                        a_new[i][j] = (pivot * a[i][j] - a[i][v] * a[u][j]) / pivot
            c_new = np.copy(c)
            for j in range(n_prime):
                if j == v:
                    c_new[j] = round(-c[j] / pivot, 6)
                else:
                    c_new[j] = round((pivot * c[j] - c[v] * a[u][j]) / pivot, 6)
            b_new = np.copy(b)
            for j in range(m_prime):
                if j == u:
                    b_new[j] = b[j] / pivot
                else:
                    b_new[j] = (pivot * b[j] - a[j][v] * b[u]) / pivot
            a = np.copy(a_new)
            c = np.copy(c_new)
            b = np.copy(b_new)
            temp2 = cb[u]
            cb[u] = cn[v]
            cn[v] = temp2
            temp1 = list_of_nonbasic_variables[v]
            list_of_nonbasic_variables[v] = list_of_basic_variables[u]
            list_of_basic_variables[u] = temp1
            for i in range(m_prime):
                s = list_of_basic_variables[i]
                x[int(s[1]) - 1] = b[i]
            for i in range(n_prime):
                s = list_of_nonbasic_variables[i]
                x[int(s[1]) - 1] = 0
            x_2 = np.copy(x)
            print_table(a, c, d, b, c, list_of_nonbasic_variables, list_of_basic_variables, x,
                        optimization)
            print("There are infinitely many solutions of the form: \u03BB", x_1, " + (1-\u03BB)", x_2)
            return
        else:
            print("The value of most negative c is", -cv, " Corresponding to column", v + 1)
            ratios = np.empty((m_prime,))
            u = 0
            pivot = -1
            min_ratio = INFINITY
            for i in range(m_prime):
                if a[i][v] == 0:
                    continue
                ratios[i] = b[i] / a[i][v]
                if min_ratio > ratios[i] >= 0:
                    u = i
                    min_ratio = ratios[i]
                    pivot = a[u][v]
            print("The ratios are for corresponding column", ratios)
            print("The minimum ratio is:", min_ratio)
            print("The pivot element is ", pivot, " and corresponding coordinates(1 based indexing) is", u + 1, " ",
                  v + 1)
            if min_ratio == INFINITY:
                print("-" * 60)
                print("The problem is unbounded")
                print("The value of objective function is:", val_objective_function(c, x, d, optimization))
                print("The values for x are:", x)
                return
            a_new = np.empty((m_prime, n_prime))
            for i in range(m_prime):
                for j in range(n_prime):
                    if i == u and j == v:
                        a_new[i][j] = 1 / a[i][j]
                    elif i == u:
                        a_new[i][j] = a[i][j] / pivot
                    elif j == v:
                        a_new[i][j] = -a[i][j] / pivot
                    else:
                        a_new[i][j] = (pivot * a[i][j] - a[i][v] * a[u][j]) / pivot
            c_new = np.copy(c)
            for j in range(n_prime):
                if j == v:
                    c_new[j] = round(-c[j] / pivot, 6)
                else:
                    c_new[j] = round((pivot * c[j] - c[v] * a[u][j]) / pivot, 6)
            b_new = np.copy(b)
            for j in range(m_prime):
                if j == u:
                    b_new[j] = b[j] / pivot
                else:
                    b_new[j] = (pivot * b[j] - a[j][v] * b[u]) / pivot
            a = np.copy(a_new)
            c = np.copy(c_new)
            b = np.copy(b_new)
            temp2 = cb[u]
            cb[u] = cn[v]
            cn[v] = temp2
            temp1 = list_of_nonbasic_variables[v]
            list_of_nonbasic_variables[v] = list_of_basic_variables[u]
            list_of_basic_variables[u] = temp1
            for i in range(m_prime):
                s = list_of_basic_variables[i]
                x[int(s[1]) - 1] = b[i]
            for i in range(n_prime):
                s = list_of_nonbasic_variables[i]
                x[int(s[1]) - 1] = 0


if __name__ == '__main__':
    # t = int(input("Enter the number of test cases:"))
    t = 1
    M = 10000
    for i in range(t):
        print("*" * 60)
        # print("This is the solution to the testcase number ", i + 1)
        n, m, a, c_original, d, b, s, optimization = take_input()
        # phase one
        a, b, cb, cn, c, x, list_of_nonbasic_variables, list_of_basic_variables, list_of_artificial_variables, list_of_slack_variables, list_of_surplus_variables = print_standard_form_phase_one(
            n, m, a, d, b, s)
        # phase one simplex
        a, b, c, x, ite, code = phase_one_simplex(a, b, cb, cn, c, d, x, list_of_nonbasic_variables, list_of_basic_variables,
                                    list_of_artificial_variables, list_of_slack_variables, list_of_surplus_variables)
        if code == 0:
            print("There is no Phase two for the problem")
        elif code==1:
            # phase two
            a, b, c, cn, cb, list_of_basic_variables, list_of_nonbasic_variables, list_of_surplus_variables, list_of_slack_variables, list_of_artificial_variables = print_standard_form_phase_two(
                a, b, c, cb, cn, c_original, x, list_of_basic_variables,
                list_of_nonbasic_variables, list_of_artificial_variables, list_of_slack_variables,
                list_of_surplus_variables)
            # phase two simplex
            phase_two_simplex(a, c, cn, cb, d, b, list_of_nonbasic_variables,
                              list_of_basic_variables,
                              x, optimization, len(list_of_surplus_variables), len(list_of_slack_variables),
                              len(list_of_artificial_variables), ite)
