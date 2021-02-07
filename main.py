import numpy as np
import png
import numpy.linalg as npl


def output_vec(vec, file):

    def vec_to_matrix(v):
        matrix = []
        for j in range(27):
            matrix.append(v[j * 28:(j + 1) * 28])
        return matrix

    accum = []

    for i in vec:
        for j in vec_to_matrix(i):
            accum.append(j)

    png.from_array(accum, 'L').save(f'images/{file}')


def get_vectors():
    vectors = np.genfromtxt('mnist_train.csv', delimiter=',', dtype=np.uint8)
    new_vectors = []

    # First element is the number the list represents
    for i in range(len(vectors)):
        new_vectors.append(np.delete(vectors[i], 0, axis=0))

    return np.array(new_vectors)


def kmeans(vectors, k, max_iterations=10, tolerance=1e-4):
    num_vectors = len(vectors)
    dimensions = len(vectors[0])
    distances = np.zeros(num_vectors)
    jPrev = np.Infinity  # placeholder for ending condition, if prev j and current j are equal, finish
    reps = [np.zeros(dimensions) for j in range(k)]
    assignment = [np.random.randint(k) for i in range(num_vectors)]

    for itr in range(max_iterations):
        for j in range(k):
            group = [i for i in range(num_vectors) if assignment[i] == j]
            reps[j] = sum(vectors[group] / len(group))
        for i in range(num_vectors):
            # (distance and index):
            (distances[i], assignment[i]) = np.amin([npl.norm(vectors[i] - reps[j]) for j in range(k)]), [
                npl.norm(vectors[i] - reps[j]) for j in range(k)].index(
                np.amin([npl.norm(vectors[i] - reps[j]) for j in range(k)]))
        J = (npl.norm(distances) ** 2) / num_vectors
        print("Iteration " + str(itr) + ": Jclust = " + str(J) + ".")
        if (itr > 1) and (abs(J - jPrev) < (tolerance * J)):
            return assignment, reps

        jPrev = J

    return assignment, reps


k = input("input k: ")
f = input("output file: ")
vecs = get_vectors()
asign, rep = kmeans(vecs, int(k))

rep_int = []
for i in range(len(rep)):
    row = []
    for j in range(len(rep[0])):
        row.append(int(rep[i][j]))
    rep_int.append(row)

output_vec(rep_int, f)
