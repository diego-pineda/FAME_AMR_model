def reduce_matrix(nodes, timesteps, matrix, fraction_nodes, fraction_timesteps):
    new_matrix = np.ones((int(timesteps / fraction_timesteps)+1, int(nodes / fraction_nodes)+1))
    ii = 0
    for i in range(timesteps+1):
        jj = 0
        if i % fraction_timesteps == 0:
            for j in range(nodes+1):
                if j % fraction_nodes == 0:
                    new_matrix[ii, jj] = matrix[i, j]
                    jj = jj + 1
            ii = ii + 1
    return new_matrix