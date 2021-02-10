import numpy as np
import pandas as pd
from itertools import combinations

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


def print_standard_form(n, m, a, c, d, b, s):
    obj_func = ""
    var = []
    a_f = []
    for i, c_prime in enumerate(c):
        obj_func += str(c_prime) + "x" + str(i + 1)
        obj_func += " + "
        var.append("x" + str(i + 1))
    size = len(c)
    obj_func += str(d)
    print("The objective function to maximize is: \n" + obj_func)
    print("\nThe standard form of the constraints using slack/surplus variables is:")
    k = n + 1
    for j, row in enumerate(a):
        constraint = ""
        for i, a_prime in enumerate(row):
            constraint += str(a_prime) + "x" + str(i + 1) + " + " * (i != len(row) - 1)
        if s[j] == "=" or s[j] == "<=":
            constraint += " + x" + str(k)
            k += 1
        else:
            constraint += " - x" + str(k) + " + x" + str(k + 1)
            k += 2
        constraint += " = " + str(b[j])
        print(constraint)
    lee = n + 1
    for j, row in enumerate(a):
        a_final = []
        for alpha, a_prime in enumerate(row):
            a_final.append(a_prime)
        for alpha in range(n, lee - 1):
            a_final.append(0)
        if s[j] == "=" or s[j] == "<=":
            a_final.append(1)
            lee += 1
        else:
            a_final.append(-1)
            a_final.append(1)
            lee += 2
        while len(a_final) < k - 1:
            a_final.append(0)
        a_f.append(a_final)
    while len(c) < k - 1:
        c.append(0)
    return np.array(a_f), c


def print_table(a_prime, c_prime, d_prime, b_prime, c, list_of_basic_variables, list_of_non_basic_variables, x):
    l_ = []
    for basic in list_of_basic_variables:
        l_.append(basic + '=' + str(x[int(basic[1]) - 1]))
    print("Non Basic Variables = ", l_)
    l_ = []
    for non_basic in list_of_non_basic_variables:
        l_.append(non_basic + '=' + str(x[int(non_basic[1]) - 1]))
    print("Basic Variables = ", l_)
    print(a_prime)
    print(b_prime)
    print(-c_prime)
    print("So the optimal value of objective function is:", round(val_objective_function(c, x, d_prime), 5))


def equation_solver(a, b):
    return np.matmul(np.linalg.inv(a),b)


def linear_solver(a, b, c):
    li = [i for i in range(a.shape[1])]
    for item in combinations(li, a.shape[0]):
        print(item)
        arr = np.array([a[:, items] for items in item]).transpose()
        # print(arr)
        try:
            ans = equation_solver(arr, b)
            li_ = []
            for i, items in enumerate(item):
                li_.append("x" + str(items+1) + " = " + str(ans[i]))
            print("The answer is: ", li_)
            x = np.zeros((a.shape[1],))
            for i, items in enumerate(item):
                x[items] = ans[i]
            d = 0
            print(x)
            print(c)
            print("The objective function is", val_objective_function(c, x, d))
            if np.min(ans)>=0:
                print("The Solution is Feasible")
            else:
                print("The Solution is Not Feasible")
        except np.linalg.LinAlgError:
            li_ = []
            for i, items in enumerate(item):
                li_.append("x" + str(items))
            print("The answer after considering ", li_, " is unbounded solution")


if __name__ == '__main__':
    # t = int(input("Enter the number of test cases:"))
    t = 1
    M = 10000
    for i in range(t):
        print("*" * 60)
        print("This is the solution to the testcase number ", i + 1)
        n, m, a, c, d, b, s = take_input()
        # Question 1
        a_f, c = print_standard_form(n, m, a, c, d, b, s)
        print(a_f, c)
        # simplex method
        linear_solver(a_f, b, c)
