"""
This code is written by Aditya Girish Pawate 18MA20054
"""
import numpy as np
import math
import pandas as pd
import warnings

def take_input():
    """
    :return: takes the input values
    """
    # print("Enter the number of strategies for player 1")
    n = int(input())
    # print("Enter the number of strategies for player 2")
    m = int(input())
    # print("Enter the pay-off matrix")
    a = []
    for i in range(n):
        b = [float(i) for i in input().split()]
        a.append(b)
    return np.array(a)


def check_stability(a):
    """
    :param a: the pay off matrix
    :return: whether the problem is stable
    """
    n = a.shape[0]
    m = a.shape[1]
    max_min = max([min([a[i][j] for j in range(m)]) for i in range(n)])
    min_max = min([max([a[i][j] for i in range(n)]) for j in range(m)])
    max_min_pt = np.argmax([min([a[i][j] for j in range(m)]) for i in range(n)])
    min_max_pt = np.argmin([max([a[i][j] for i in range(n)]) for j in range(m)])
    if max_min == min_max:
        print("The problem is stable and the saddle point is", max_min)
        str_1 = np.zeros((n,))
        str_2 = np.zeros((m,))
        str_1[min_max_pt] = 1
        str_2[max_min_pt] = 1
        print("The probability of Strategies for player 1 is", str_1)
        print("The probability of Strategies for player 2 is", str_2)
        print("The value of the game is:", max_min)
    else:
        print("The problem is unstable further analysis is required to find the mixed strategies")
    return max_min == min_max


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


def find_strategies(a, min_ele):
    n = a.shape[0]
    m = a.shape[1]
    a = a.tolist()
    a_new = []
    for item in a:
        item.append("<=")
        a_new.append(item)
    c = np.array([1]*m)
    d = 0
    b = np.array([1]*n)
    Zj, df = simplex_method(a_new, b, n, m, c, "max")
    A_strategy = Zj.values[0][-n-1:-1]
    B_strategy_var = df['basic_var'].tolist()
    B_strategy = df['sol'].tolist()
    print("The optimal solutions of strategies for A are:", A_strategy)
    print("The optimal solutions of strategies for B for ",  B_strategy)

    print("The probability of strategies for A are:", A_strategy/Zj.values[0][-1])
    print("The probability of strategies for B for variables", B_strategy_var, " is ", B_strategy/Zj.values[0][-1])
    print("The rest probabilities of strategies for B are 0")

    print("The optimal value of the game is V = ", 1/Zj.values[0][-1] - min_ele)
    return


def val_objective_function(c, x, d):
    return np.dot(c, x) + d


def make_positive(a):
    min_element = np.min(a)
    if min_element <= 0:
        return a + abs(min_element) + 1, abs(min_element)+1
    else:
        return a, 0


def convert_to_maximization_type(a, b, Coeff, m, n, objective):
    if objective != 'max':
        Coeff = [-Coeff[i] for i in range(n)]
    for i in range(m):
        if a[i][-1] != '<=':
            for j in range(n):
                a[i][j] = -a[i][j]
            b[i] = -b[i]
    return a, b, Coeff, m, n


def initialize_simplex_table(a, b, m, n, Coeff, objective):
    convert_dict = {'x' + str(i + 1): float for i in range(n)}
    jconvert_dict = {'x' + str(i + 1): float for i in range(n)}
    a, b, Coeff, m, n = convert_to_maximization_type(a, b, Coeff, m, n, objective)
    Cj_dict = {'x' + str(i + 1): [Coeff[i]] for i in range(n)}
    Zj_dict = {'x' + str(i + 1): [0] for i in range(n)}
    for i in range(m):
        Cj_dict['s' + str(i + 1)] = [0]
        Zj_dict['s' + str(i + 1)] = [0]
        convert_dict['s' + str(i + 1)] = float
        jconvert_dict['s' + str(i + 1)] = float
    Cj_dict['sol'] = 0
    Zj_dict['sol'] = 0
    convert_dict['sol'] = float
    jconvert_dict['sol'] = float
    # Cj_dict.extend({0 for i in range(m)])
    # Zj=[0 for i in range(m+n)]
    data = {'CBi': [0 for i in range(m)]}
    convert_dict['CBi'] = float
    data['basic_var'] = ['s' + str(i + 1) for i in range(m)]
    convert_dict['basic_var'] = str
    for i in range(n):
        data['x' + str(i + 1)] = [a[j][i] for j in range(m)]
    # augmenting data with slack variables
    for i in range(m):
        data['s' + str(i + 1)] = [float(i == j) for j in range(m)]
    data['sol'] = [b[j] for j in range(m)]
    data['ratio'] = [0 for j in range(m)]
    convert_dict['ratio'] = float
    df = pd.DataFrame(data, index=[str(i + 1) for i in range(m)])
    Zj = pd.DataFrame(Zj_dict, index=['Zj'])
    Cj = pd.DataFrame(Cj_dict, index=['Cj'])

    df = df.astype(convert_dict)

    Zj = Zj.astype(jconvert_dict)
    Cj = Cj.astype(jconvert_dict)
    return df, Cj, Zj


def print_table(df, Cj, Zj):
    print("simplex table:")
    frames = [df, Cj, Zj]
    result = pd.concat(frames)

    print(result)



def Zj_Cj(Cj, Zj, m, n):
    l = ['x' + str(i + 1) for i in range(n)]
    l.extend(['s' + str(i + 1) for i in range(m)])
    return [Zj[i]['Zj'] - Cj[i]['Cj'] for i in l]


def col_name(i, n):
    if (i + 1) <= n:
        return 'x' + str(i + 1)
    else:
        return 's' + str(i + 1 - n)


def get_key_column(zj_cj, m, n):
    val_min = zj_cj[0]
    idx_min = 'x1'
    for i in range(m + n):
        if val_min > zj_cj[i]:
            val_min = zj_cj[i]
            idx_min = col_name(i, n)

    return idx_min


def get_key_row(df, key_col, m):
    df['ratio'] = df['sol'] / df[key_col]
    val_min = 999999999
    idx_min = 0
    for i in range(m):
        if val_min > df['ratio'][i] >= 0:
            val_min = df['ratio'][i]
            idx_min = i
    return df, str(idx_min + 1), val_min


def modify_table(key_row, key_col, Cj, Zj, df, m,n):
    warnings.filterwarnings("ignore")
    df_new = df.copy(deep=True)
    Zj_new = Zj.copy(deep=True)
    col_len = len(df[key_col])
    row_len = len(df.loc[key_row])
    p = key_elem = df[key_col][key_row]
    entering_var = key_col
    leaving_var = df['basic_var'][key_row]
    print("Entering variable is ", entering_var)
    print("Leaving variable is ", leaving_var)
    df_new['basic_var'][key_row] = entering_var

    col = ['x' + str(i + 1) for i in range(n)]
    col.extend(['s' + str(i + 1) for i in range(m)])
    col.append('sol')
    row = list(df.index)

    for j in col:
        r = df[j][key_row]
        for i in row:
            if j == key_col:
                df_new[j][i] = (-df[j][i]) / p
            elif i == key_row:
                df_new[j][i] = df[j][i] / p
            else:
                s = df[j][i]
                q = df[key_col][i]
                df_new[j][i] = (p * s - q * r) / p
    # print(1/p)
    df_new[key_col][key_row] = 1 / p
    # df_new[key_col]=df[]
    df_new[leaving_var] = df_new[entering_var]
    df_new[entering_var] = 0
    df_new[entering_var][key_row] = 1

    # swap the coefficients value
    df_new['CBi'][key_row] = Cj[key_col]['Cj']

    # modify zj
    for j in col:
        Zj_new[j]['Zj'] = sum([df_new[j][i] * df_new['CBi'][i] for i in row])
        # Zj_new[j]['Zj']=16.0*0.05
    return df_new, Cj, Zj_new


def is_optimal(zj_cj):
    val = min(zj_cj)
    if val >= 0:
        print("Optimality is reached")
        return True
    else:
        print("Lets proceed to next iteration")
        return False


def simplex_method(a, b, m, n, Coeff, objective):
    df, Cj, Zj = initialize_simplex_table(a, b, m, n, Coeff, objective)
    zj_cj = Zj_Cj(Cj, Zj, m, n)
    key_col = get_key_column(zj_cj, m, n)
    df, key_row, min_ratio = get_key_row(df, key_col, m)
    print_table(df, Cj, Zj)
    i = 0
    while not is_optimal(zj_cj) and i < 10:
        print("_______________________________________________________")
        if df[key_col][key_row] == 0:
            print("Problem is unbounded")
            return
        df_new, Cj, Zj_new = modify_table(key_row, key_col, Cj, Zj, df, m, n)

        df = df_new
        Zj = Zj_new

        if objective == "max":
            print("value of objective function is ", Zj['sol']['Zj'])
        else:
            print("value of objective function is ", -Zj['sol']['Zj'])

        zj_cj = Zj_Cj(Cj, Zj, m, n)
        key_col = get_key_column(zj_cj, m, n)
        df, key_row, min_ratio = get_key_row(df, key_col, m)
        print_table(df, Cj, Zj)
        print(df['sol'])
        print(Zj)
        i += 1
    return Zj, df


if __name__ == '__main__':
    pay_off = take_input()
    if not (check_stability(pay_off)):
        pay_off, min_ele = make_positive(pay_off)
        print(pay_off)
        find_strategies(pay_off, min_ele)
