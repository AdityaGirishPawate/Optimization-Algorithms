import numpy as np


def take_input():
    n = int(input())
    matrix = []
    for i in range(n):
        row = [float(i) for i in input().split()]
        matrix.append(row)
    if len(matrix) > len(matrix[0]):
        for cnt in range(len(matrix) - len(matrix[0])):
            for i in matrix:
                i.append(0)
    elif len(matrix) < len(matrix[0]):
        for cnt in range(len(matrix[0]) - len(matrix)):
            matrix.append([0] * len(matrix[0]))
    assert len(matrix) == len(matrix[0])
    return matrix


def cost(matrix, path):
    cost = 0
    for i in range(len(matrix)):
        print("The worker", i + 1, "and the Job ", path[i] + 1)
        print("Cost for  this assignment is:", matrix[i][path[i]])
        cost += matrix[i][path[i]]
    print("The total Minimum cost for assignment is", cost)


def row_reduction(m):
    """
    step 1: row reduction
    """
    assert isinstance(m, np.matrixlib.defmatrix.matrix)
    print("step 1: row reduction")
    print("The minimum elements are")
    print(np.min(m, axis=1))
    print("The matrix after row reduction is:")
    print(m - np.min(m, axis=1))
    return m - np.min(m, axis=1)


def col_reduction(m):
    """
    step 2: col reduction
    """
    assert isinstance(m, np.matrixlib.defmatrix.matrix)
    print("step 2: col reduction")
    print("The minimum elements are")
    print(np.min(m, axis=0))
    print("The matrix after column reduction is:")
    print(m - np.min(m, axis=0))
    return m - np.min(m, axis=0)


def find_filter(m, max_num_filter=6):
    """
    Step 3 find filter 
    """
    paths = {'path': []}  # Name binding

    def get_redundant_index(v):
        n = len(v)
        occurrence_vector = [v.count(i) for i in range(n)]
        return max(occurrence_vector) - 1 + occurrence_vector.count(0) * 1.0 / n

    def search_for_paths(m, tracing):
        n_cols = m.shape[1]
        tracing_number = len(tracing)
        # print tracing
        if tracing_number < n_cols:
            col_index_arranged = list(set(range(n_cols)) - set(tracing)) + list(set(tracing))
            for j in col_index_arranged:
                if m[tracing_number, j] == 0:
                    # More scouts are launched only if some place in the paths' vector is available
                    # Rearranging indexes may guarantee that a filter with 0 redundancy index is stored in the
                    # the first explorations, when available.
                    if len(paths['path']) < 2 * max_num_filter:
                        search_for_paths(m, tracing + [j])
        elif tracing_number == n_cols:
            path_recorder(tracing)

    def path_recorder(v):
        ri = get_redundant_index(v)
        paths['path'] = paths['path'] + [v] + [ri]

    first_zero = True
    for j in range(m.shape[1]):
        if m[0, j] == 0:
            if first_zero:
                # reset to [] the list of paths
                paths['path'] = []
                first_zero = False
            # print 'A search_for_paths goes in mission'
            search_for_paths(m, [j])

    if len(paths['path']) == 0:
        raise TypeError('Input matrix has no 0-filters.')
    return [m, paths['path']]


def is_optimal(m, paths_):
    """
    :param m: cost matrix
    :param paths_: list of 0-filter followed by its index of redundancy
    as returned by find_filter
    :return: M again untouched, followed by the list of $0$-filter with
    minimal index of redundancy, and with a flag, True if the minimal index
    is 0 and so we have already our solution, False otherwise.
    """
    min_redundancy = np.min(paths_[1::2])
    filtered_paths = [paths_[i] for i in list(range(len(paths_)))[::2] \
                      if paths_[i + 1] == min_redundancy]
    if min_redundancy == 0:
        flag = True
    else:
        flag = False
    return [m, filtered_paths, flag]


def covering_segments_searcher(m, min_redundancy_filter):
    """
     step 5, auxiliary function.
     Returns the positions of horizontal and vertical covering segments
    """
    # (A)
    n_rows = m.shape[0]
    n_cols = m.shape[1]
    marked_row = [0] * n_rows
    marked_col = [0] * n_cols
    # (B)
    occurrence_vector = [min_redundancy_filter.count(i) for i in range(n_rows)]
    for pos in range(n_rows):
        if occurrence_vector[pos] > 1:
            duplicates_pos = [k for k in range(n_rows) if min_redundancy_filter[k] == pos][1:]
            for j in duplicates_pos:
                marked_row[j] = 1
    # (C)
    flag_mark = 1
    while flag_mark != 0:
        flag_mark = 0
        # (C-1)
        for i in range(n_rows):
            if marked_row[i] == 1:
                for j in range(n_cols):
                    if m[i, j] == 0 and marked_col[j] != 1:
                        marked_col[j] = 1
                        flag_mark += flag_mark
        # (C-2)
        for j in range(n_cols):
            if marked_col[j] == 1:
                for i in range(n_rows):
                    if m[i, j] == 0 and marked_row[i] != 1 and min_redundancy_filter[i] == j:
                        marked_row[i] = 1
                        flag_mark += 1
    # (D)
    covered_row = [(i + 1) % 2 for i in marked_row]
    covered_col = marked_col

    return covered_row, covered_col


def mix_matrix(m, filtered_paths):
    n_rows = m.shape[0]
    n_cols = m.shape[1]
    min_redundancy_filter = filtered_paths[0]
    # (1)
    [cov_row, cov_col] = covering_segments_searcher(m, min_redundancy_filter)
    # (2)
    zero_pos_in_cov_row = [i for i in range(n_rows) if cov_row[i] == 0]
    zero_pos_in_cov_col = [j for j in range(n_cols) if cov_col[j] == 0]
    uncovered_elements = [m[i, j] for i in zero_pos_in_cov_row for j in zero_pos_in_cov_col]
    if not uncovered_elements:  # uncovered_elements == []
        raise EnvironmentError('Not enough filters has been considered, '
                               'set an higher max_num_filter parameter!')

    min_val = min(uncovered_elements)
    # (3)
    for i in range(n_rows):
        for j in range(n_cols):
            if cov_row[i] == 0 == cov_col[j]:
                m[i, j] -= min_val
            elif cov_row[i] == 1 == cov_col[j]:
                m[i, j] += 2 * min_val
    return m


def hungarian(m, max_num_filter=100):
    cont = 0
    max_loop = max(m.shape[0], m.shape[1])
    s = row_reduction(m)
    s = col_reduction(s)
    [s, paths] = find_filter(s, max_num_filter=max_num_filter)
    [s, filtered_paths, flag] = is_optimal(s, paths)
    while not flag and cont < max_loop:
        s = mix_matrix(s, filtered_paths)
        [s, paths] = find_filter(s, max_num_filter=max_num_filter)
        [s, filtered_paths, flag] = is_optimal(s, paths)
        cont += 1
    l = []
    for _paths in filtered_paths:
        l_ = []
        for i in range(len(m)):
            l_.append(("Worker:" + str(i), "Job:" + str(_paths[i])))
        l.append(l_)
    print("Optimal Assignment is/are")
    for i, l_ in enumerate(l):
        print("Solution", i + 1)
        print(l_)
    return filtered_paths


if __name__ == '__main__':
    m = take_input()
    path = hungarian(np.matrix(m), 10)
    cost(m, path[0])
