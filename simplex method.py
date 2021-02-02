import numpy as np

INFINITY = 10e9


def take_input():
    # print("Enter the number of variables in the objective function")
    n = int(input())
    # print("Enter the number of constraints")
    m = int(input())
    # print("Enter the coefficients of the Objective Function")
    c = [float(i) for i in input().split(" ")]
    c = np.array(c)
    # print("Enter the value of the constant in the objective function")
    d = float(input())
    # print("Enter the matrix A which is the coefficient of the constraints row by row")
    a = []
    for i in range(m):
        a.append([float(j) for j in input().split(" ")])
    a = np.array(a)
    # print("Enter the constants/RHS bi for the constraint equations")
    b = [float(i) for i in input().split(" ")]
    b = np.array(b)
    return n, m, a, c, d, b


def val_objective_function(c, x, d):
    return np.dot(c, x) + d


def print_standard_form(n, m, a, c, d, b):
    obj_func = ""
    for i, c_prime in enumerate(c):
        obj_func += str(c_prime) + "x" + str(i)
        obj_func += " + "
    obj_func += str(d)
    print("The objective function to maximize is: \n" + obj_func)
    print("\nThe standard form of the constraints using slack/surplus variables is:")
    for j, row in enumerate(a):
        constraint = ""
        for i, a_prime in enumerate(row):
            constraint += str(a_prime) + "X" + str(i + 1) + " + "
        constraint += "Z" + str(j + 1)
        constraint += " = " + str(b[j])
        print(constraint)


def simplex(n_prime, m_prime, a_prime, c_prime, d_prime, b_prime):
    c = np.copy(c_prime)
    x = np.zeros((n_prime,))
    z = np.copy(b_prime)
    ite = 0
    list_of_basic_variables = {i + 1: "z" + str(i + 1) for i in range(m_prime)}
    list_of_non_basic_variables = {i + 1: "x" + str(i + 1) for i in range(n_prime)}
    while True:
        ite += 1
        print("-" * 60)
        print("Iteration ", ite)
        print("The value of Objective function in this iteration is ", val_objective_function(c, x, d_prime))
        print("The values for x are:")
        l_ = []
        for i in range(len(x)):
            l_.append("x" + str(i + 1) + '=' + str(x[i]))
        print(l_)
        print("This is the list of all the basic variables are ", list_of_basic_variables)
        print("This is the list of all non-basic variables are ", list_of_non_basic_variables)
        print("The matrix form of A is")
        print(a_prime)
        print("The negative values of C are")
        print(-c_prime)
        print("The values of basic solutions X_b are")
        print(b_prime)
        v = np.argmax(c_prime)
        cv = c_prime[v]
        if cv <= 0:
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
        print("The value of most negative c_prime is", -cv, " Corresponding to column", v+1)
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
            print("The problem is unbounded")
            print("The value of objective function is:", val_objective_function(c_prime, x, d_prime))
            print("The values for x are:", x)
            return
        print("The ratios are for corresponding column", ratios)
        print("The minimum ratio is:", min_ratio)
        print("The pivot element is ", pivot, " and corresponding coordinates(1 based indexing) is", u+1, " ", v+1)
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
                c_new[j] = -c_prime[j] / pivot
            else:
                c_new[j] = (pivot * c_prime[j] - c_prime[v] * a_prime[u][j]) / pivot
        b_new = np.copy(b_prime)
        for j in range(m_prime):
            if j == u:
                b_new[j] = b_prime[j] / pivot
            else:
                b_new[j] = (pivot * b_prime[j] - a_prime[j][v] * b_prime[u]) / pivot
        a_prime = np.copy(a_new)
        c_prime = np.copy(c_new)
        b_prime = np.copy(b_new)
        list_of_basic_variables[u + 1] = "x" + str(v + 1)
        list_of_non_basic_variables[v + 1] = "z" + str(u + 1)
        for i in range(m_prime):
            s = list_of_basic_variables[i + 1]
            if s[0] == 'z':
                z[int(s[1]) - 1] = b_prime[i]
            elif s[0] == 'x':
                x[int(s[1]) - 1] = b_prime[i]


if __name__ == '__main__':
    t = int(input("Enter the number of test cases:"))
    for i in range(t):
        print("*" * 60)
        print("This is the solution to the testcase number ", i + 1)
        n, m, a, c, d, b = take_input()
        # Question 1
        print_standard_form(n, m, a, c, d, b)
        # simplex method
        simplex(n, m, a, c, d, b)
