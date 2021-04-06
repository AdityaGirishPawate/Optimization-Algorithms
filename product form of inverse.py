import numpy as np
from math import *


def take_input():
    # Enter the number of rows and number of columns respectively
    (n, m) = (int(i) for i in input().split())
    b = np.zeros((n, m))
    # Enter the matrix b
    for i in range(n):
        b[i] = np.array([int(i) for i in input().split()])
    bc = np.zeros((n, m))
    # Enter the matrix bc
    for i in range(n):
        bc[i] = np.array([int(i) for i in input().split()])
    return b, bc


def calculate_inverse(b, b_inverse, bc):
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
        print("The two matrices are differing by more than one column")
        return
    if number_of_different_columns == 0:
        print("The two given matrices re the same")
        return
    # compute the e vector
    e = np.matmul(b_inverse, cr)
    print("This is the e vector:", e)
    # compute eta
    eta = -e/e[r]
    eta[r] = 1/e[r]
    print("This is the value of eta", eta)
    # computing Er
    er = np.copy(b)
    er[:, r] = eta
    # Finally compute the inverse
    bc_inverse = np.matmul(er, b_inverse)
    return bc_inverse


if __name__ == '__main__':
    # take the input
    b, bc = take_input()
    bc_inverse = calculate_inverse(b, b, bc)
    print(bc_inverse)
    
