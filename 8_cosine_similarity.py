import numpy as np

def cosine_similarity(a,b):
    a = np.asarray(a, dtype=float) #converts arr into float arr
    b = np.asarray(b,dtype=float)

    if a.shape != b.shape:
        raise ValueError("Vectors must have the same shape")

    a_norm = np.linalg.norm(a); # gets the norm of the arr , norm = square root of sum of all squared numbers of arr
    b_norm = np.linalg.norm(b);

    if a_norm == 0 or b_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")

    return float(np.dot(a,b) / (a_norm * b_norm))

if __name__ == "__main__":
    a = np.array([1.0,0.0])
    b = np.array([2.0,0.5])

    print(cosine_similarity(a,b))