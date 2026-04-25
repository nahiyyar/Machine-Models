import numpy as np 
import matplotlib.pyplot as plt

def pca(X, n_components=2):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]
    
    X_reduced = X_centered @ top_eigenvectors
    return X_reduced, top_eigenvectors, np.mean(X, axis=0)

if __name__ == "__main__":

    np.random.seed(42)
    x = np.random.rand(100)
    y = 2 * x + np.random.normal(0, 0.1, 100)
    X = np.column_stack((x, y))
    X_reduced, components, mean = pca(X, n_components=1)

    X_projected = X_reduced @ components.T + mean

    plt.scatter(X[:, 0], X[:, 1], label="Original")
    plt.scatter(X_projected[:, 0], X_projected[:, 1], label="Projected")

    plt.legend()
    plt.show()
