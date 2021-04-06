"""
Revised Simplex Method
Lab -7
Aditya Girish Pawate
18MA20054
"""


import numpy as np
import pandas as pd
from math import *


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
    # print("Enter the matrix A which is the coefficient of the constraints row by row")
    a = []
    for i in range(m):
        a.append([float(j) for j in input().split(" ")])
    # print("Enter the constants/RHS bi for the constraint equations")
    b = [float(i) for i in input().split(" ")]
    # print("Enter the equation or in-equation type fo the variable")
    s = input().split(" ")
    for i, ite in enumerate(s):
        if ite == '=':
            print("Equality type constraint not allowed for revised simplex method")
            exit(-1)
        if ite == '>=':
            for j in range(n):
                a[i][j] *= -1
            b[i] *= -1
            s[i] = '<='
    return n, m, np.array(a), np.array(c), np.array(b), s, optimization


def get_cb(cv, list_of_basic_variables):
    cb = np.empty((len(list_of_basic_variables)))
    for i, item in enumerate(list_of_basic_variables):
        cb[i] = float(cv[item])
    return np.array(cb)


def print_initial_table(n, m, a, c, b):
    print("*" * 75)
    print("Initial Extended Simplex Table")
    list_of_variables = []
    list_of_nonbasic_variables = []
    list_of_basic_variables = []
    map_of_columns = {}
    cv = {}
    identity = np.identity(m)
    for i in range(n):
        list_of_variables.append("x" + str(i + 1))
        list_of_nonbasic_variables.append("x" + str(i + 1))
        map_of_columns["x" + str(i + 1)] = a[:, i]
        cv["x" + str(i + 1)] = c[i]
    for i in range(m):
        list_of_variables.append("x" + str(i + 1 + n))
        list_of_basic_variables.append("x" + str(i + 1 + n))
        map_of_columns["x" + str(i + 1 + n)] = identity[:, i]
        cv["x" + str(i + 1 + n)] = 0.0
    cb = get_cb(cv, list_of_basic_variables)
    print("cv: ", cv)
    df = pd.DataFrame(columns=['cb', 'b\\v'] + list_of_variables + ['b'])
    df['cb'] = cb
    df['b\\v'] = list_of_basic_variables
    for i in range(len(list_of_variables)):
        df[list_of_variables[i]] = map_of_columns[list_of_variables[i]]
    df['b'] = b
    print(df)
    zj_cj = np.empty((len(list_of_variables)))
    for i, item in enumerate(list_of_variables):
        zj_cj[i] = np.dot(cb, map_of_columns[item]) - cv[list_of_variables[i]]
    # print(cb, map_of_columns)
    print("zj - cj: ", zj_cj)
    return cv, cb, identity, list_of_basic_variables, list_of_nonbasic_variables, map_of_columns


def calculate_obj(cb, Xb):
    return np.dot(cb, Xb)


def revised_simplex(cv, b, B, B_inverse, list_of_basic_variables, list_of_nonbasic_variables, map_of_columns):
    ite = 0
    while True:
        # Print the basic and non basic variables
        print("-" * 75)
        print("Iteration ", ite + 1)
        ite += 1
        if ite >= 10:
            print("The problem is unbounded / infeasible")
            return
        print("List of Basic variables is: ", list_of_basic_variables)
        print("List of non-Basic variables is:", list_of_nonbasic_variables)
        print("The B matrix is: \n", B)
        print("The B_inverse matrix is: \n", B_inverse)
        # get the value of cb
        cb = get_cb(cv, list_of_basic_variables)
        print("The value of Cb is:", cb)
        # compute Y
        Y = np.dot(np.transpose(cb), B_inverse)
        print("The value of Y is ", Y)
        # compute zj - cj for all the non-basic variables
        zj_cj = np.empty((len(list_of_nonbasic_variables)))
        for i, item in enumerate(list_of_nonbasic_variables):
            zj_cj[i] = np.dot(Y, map_of_columns[item]) - cv[item]
        print("The vector zj-cj is ", zj_cj)
        # find the minimum (most negative) columns
        min_col = np.argmin(zj_cj)
        # calculate the value of Xb
        Xb = np.matmul(B_inverse, b)
        # Calculate the value of Z (optimization function)
        Z = calculate_obj(cb, Xb)
        if min(zj_cj) >= 0:
            print("The iterations have ended. The optimal solution is reached.")
            print("The value of Xb is ", Xb)
            print("The value of variables are:")
            for i, item in enumerate(list_of_basic_variables):
                print(item + "=" + str(round(Xb[i], 4)))
            print("The value of objective function is ", round(float(Z), 4))
            return
        print("The minimum column is ", min_col+1)
        print("The value of Xb is ", Xb)
        print("The value of objective function is ", Z)
        # calculate the value of alpha_j
        alpha_j = np.matmul(B_inverse, map_of_columns[list_of_nonbasic_variables[min_col]])
        print("The matrix alpha_j is:", alpha_j)
        min_ratio_ind = -1
        min_ratio = int(10e8)
        # find the leaving variable by finding the min_ratio
        for i in range(len(b)):
            if alpha_j[i] > 0 and min_ratio > Xb[i] / alpha_j[i]:
                min_ratio = Xb[i] / alpha_j[i]
                min_ratio_ind = i
        if alpha_j.any() <= 0 and min_ratio <= 0:
            print("The problem is unbounded / infeasible")
            return
        print("The minimum ratio is ", min_ratio)
        print("The corresponding row is ", min_ratio_ind + 1)
        entering_variable = list_of_nonbasic_variables[min_col]
        leaving_variable = list_of_basic_variables[min_ratio_ind]
        print("The entering variable is:", entering_variable)
        print("The leaving variable is:", leaving_variable)
        list_of_basic_variables[min_ratio_ind] = entering_variable
        list_of_nonbasic_variables[min_col] = leaving_variable
        B_new = np.copy(B)
        B_new[:, min_ratio_ind] = np.copy(map_of_columns[entering_variable])
        B_new_inverse = calculate_inverse(B, B_inverse, B_new)
        # print("B is : \n", B)
        # print("B new is: \n", B_new)
        # print("B new inverse is: \n", B_new_inverse)
        B = np.copy(B_new)
        B_inverse = np.copy(B_new_inverse)


def calculate_inverse(b, b_inverse, bc):
    """
    This function calculates the inverse of a matrix using product form of inverse
    :param b: the basis matrix
    :param b_inverse: inverse of basis matrix
    :param bc: new basis matrix
    :return: inverse of new basis matrix
    """
    bc_inverse = np.zeros_like(b)
    # First find the column number which is different
    number_of_different_columns = 0
    r = -1
    cr = np.zeros((b.shape[0]))
    for i in range(b.shape[1]):
        if list(b[:, i]) != list(bc[:, i]):
            r = i
            number_of_different_columns += 1
            cr = np.copy(bc[:, i])
    if number_of_different_columns > 1:
        # print("The two matrices are differing by more than one column")
        return
    if number_of_different_columns == 0:
        # print("The two given matrices re the same")
        return b_inverse
    # compute the e vector
    e = np.matmul(b_inverse, cr)
    # print("This is the e vector:", e)
    # compute eta
    eta = -e / e[r]
    eta[r] = 1 / e[r]
    # print("This is the value of eta", eta)
    # computing Er
    er = np.copy(b)
    er[:, r] = eta
    # Finally compute the inverse
    bc_inverse = np.matmul(er, b_inverse)
    assert bc_inverse.all() == np.linalg.inv(bc).all()
    return np.linalg.inv(bc)


if __name__ == '__main__':
    # take the input
    n, m, a, c, b, s, optimization = take_input()
    cv, cb, identity, list_of_basic_variables, list_of_nonbasic_variables, map_of_columns = print_initial_table(n, m, a,
                                                                                                                c, b)
    revised_simplex(cv, b, identity, identity, list_of_basic_variables, list_of_nonbasic_variables, map_of_columns)
