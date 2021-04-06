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
    for i,ite in enumerate(s):
        if(ite == '='):
            print("Equality type constraint not allowed for dual simplex method")
            exit(-1)
        if(ite == '>='):
            for j in range(n):
                a[i][j] *= -1
            b[i] *= -1
            s[i] = '<='
    return n, m, a, np.array(c), d, b, s, optimization


def val_objective_function(cb, b, d, optimization):
    if optimization == 'max':
        return np.dot(np.transpose(cb), b) + d
    else:
        return -np.dot(np.transpose(cb), b) + d


def print_standard_form(n, m, a, c, d, b, s):
    num_artificial_variables = 0
    num_surplus_variables = 0
    num_slack_variables = 0
    k = n + 1
    x = []
    add_col = []
    all_variables = []
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
        if s[j] == "<=":
            constraint += " + x" + str(k)
            list_of_basic_variables.append("x" + str(k))
            list_of_slack_variables.append("x" + str(k))
            x.append(b[j])
            temp_col = np.zeros((m,))
            temp_col[j] = 1
            add_col.append(temp_col)
            k += 1
            num_slack_variables += 1
        constraint += " = " + str(b[j])
        constraints.append(constraint)
    all_variables = list_of_nonbasic_variables + list_of_basic_variables
    cn = np.zeros((len(all_variables)))
    cb = np.zeros((len(list_of_basic_variables)))
    for i, var in enumerate(list_of_basic_variables):
        cb[i] = 0
    for i in range(len(list_of_nonbasic_variables)):
        cn[i] = c[i]
    c = np.empty((len(all_variables)))
    a = np.concatenate([a, np.array(add_col)], axis=1)
    for i in range(len(all_variables)):
        c[i] = 0
        c[i] += (np.dot(np.transpose(cb), a[:, i]) - cn[i])
    print("The objective function to maximize is")
    obj_func = ""
    for i, var in enumerate(list_of_nonbasic_variables):
        obj_func += "-" + str(-c[i]) + str(var) + " "
    print(obj_func + " + ", d)
    for constraint in constraints:
        print(constraint)
    return a, b, cb, cn, c, np.array(
        x), list_of_nonbasic_variables, list_of_basic_variables, all_variables, list_of_artificial_variables, list_of_slack_variables, list_of_surplus_variables


def print_table(a, c, d, b, cb, x, list_of_nonbasic_variables, list_of_basic_variables, optimization):
    l_ = []
    for basic in list_of_nonbasic_variables:
        l_.append(basic)
    print("Non Basic Variables = ", l_)
    l_ = []
    for non_basic in list_of_basic_variables:
        l_.append(non_basic)
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


def dual_simplex(a, c, cn, cb, d, b, list_of_nonbasic_variables, list_of_basic_variables, all_variables,
                      x, optimization, num_surplus_variables, num_slack_variables, num_artificial_variables):
    n_prime = a.shape[1]
    m_prime = a.shape[0]
    ite = 0
    while True:
        ite += 1
        if ite > 10:
            print("The table has repeated. Due to this there is infinite iterations. Hence cannot solve by dual simplex.")
            exit(0)
        print("-" * 60)
        print("Iteration ", ite)
        print_table(a, c, d, b, cb, x, list_of_nonbasic_variables, list_of_basic_variables,
                    optimization)
        print("The value of Objective function in this iteration is ",
              round(val_objective_function(cb, b, d, optimization), 5))
        print("The values for x are:")
        l_ = []
        for i in range(len(x)):
            l_.append("x" + str(i + 1) + '=' + str(x[i]))
        print(l_)
        u = np.argmin(b)
        cv = b[u]
        if cv > 0:
            if ite > num_artificial_variables:
                print("-" * 60)
                print("The iterations have ended because no negative values of Xb are present")
                print("This is the list of all the non-basic variables are ", list_of_nonbasic_variables)
                print("This is the list of all basic variables are ", list_of_basic_variables)
                print("The values for x are:")
                l_ = []
                for i in range(len(x)):
                    l_.append("x" + str(i + 1) + '=' + str(x[i]))
                print(l_)
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
                    li.append("x" + str(i + 1))
                print(li)
                print("As you can see the the value of objective function is :",
                      round(val_objective_function(cb, b, d, optimization), 5))
                return
        if cv == 0:
            x_1 = np.copy(x)
            ratios = np.empty((n_prime,))
            v = 0
            pivot = -1
            min_ratio = INFINITY
            for i in range(n_prime):
                if a[u][i] == 0:
                    continue
                ratios[i] = abs(c[i] / a[u][i])
                if min_ratio > ratios[i] > 0:
                    v = i
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
                        a_new[i][j] = 1
                    elif i == u:
                        a_new[i][j] = a[i][j] / pivot
                    elif j == v:
                        a_new[i][j] = 0
                    else:
                        a_new[i][j] = (pivot * a[i][j] - a[i][v] * a[u][j]) / pivot
            c_new = np.copy(c)
            for j in range(n_prime):
                if j == v:
                    c_new[j] = 0
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
            print_table(a, c, d, b, cb, list_of_nonbasic_variables, list_of_basic_variables, x,
                        optimization)
            print("There are infinitely many solutions of the form: \u03BB", x_1, " + (1-\u03BB)", x_2)
            return
        else:
            print("The value of most negative Xb is", cv, " Corresponding to row", u + 1)
            ratios = np.empty((n_prime,))
            v = 0
            pivot = -1
            min_ratio = INFINITY
            for i in range(n_prime):
                if a[u][i] == 0:
                    continue
                ratios[i] = abs(c[i] / a[u][i])
                if min_ratio > ratios[i] >0:
                    v = i
                    min_ratio = ratios[i]
                    pivot = a[u][v]
            print("The ratios are for corresponding row", ratios)
            print("The minimum ratio is:", min_ratio)
            print("The pivot element is ", pivot, " and corresponding coordinates(1 based indexing) is", u + 1, " ",
                  v + 1)
            if min_ratio == INFINITY:
                print("-" * 60)
                print("The problem is unbounded")
                print("The value of objective function is:", val_objective_function(cb, b, d, optimization))
                print("The values for x are:", x)
                return
            a_new = np.empty((m_prime, n_prime))
            for i in range(m_prime):
                for j in range(n_prime):
                    if i == u and j == v:
                        a_new[i][j] = 1
                    elif i == u:
                        a_new[i][j] = a[i][j] / pivot
                    elif j == v:
                        a_new[i][j] = 0
                    else:
                        a_new[i][j] = (pivot * a[i][j] - a[i][v] * a[u][j]) / pivot
            c_new = np.copy(c)
            for j in range(n_prime):
                if j == v:
                    c_new[j] = 0
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
            temp1 = all_variables[v]
            list_of_nonbasic_variables.remove(all_variables[v])
            if list_of_basic_variables[u] not in list_of_nonbasic_variables:
                list_of_nonbasic_variables.append(list_of_basic_variables[u])
            list_of_basic_variables[u] = temp1
            for i in range(len(list_of_basic_variables)):
                s = list_of_basic_variables[i]
                x[int(s[1]) - 1] = b[i]
            for i in range(len(list_of_nonbasic_variables)):
                s = list_of_nonbasic_variables[i]
                x[int(s[1]) - 1] = 0


if __name__ == '__main__':
    # t = int(input("Enter the number of test cases:"))
    t = 1
    M = 10000
    for i in range(t):
        print("*" * 60)
        # print("This is the solution to the testcase number ", i + 1)
        n, m, a, c, d, b, s, optimization = take_input()
        # print standard form
        a, b, cb, cn, c, x, list_of_nonbasic_variables, list_of_basic_variables, all_variables, list_of_artificial_variables, list_of_slack_variables, list_of_surplus_variables = print_standard_form(n, m, a, c, d, b, s)
        # dual simplex
        dual_simplex(a, c, cn, cb, d, b, list_of_nonbasic_variables,
                          list_of_basic_variables, all_variables,
                          x, optimization, len(list_of_surplus_variables), len(list_of_slack_variables),
                          len(list_of_artificial_variables))
