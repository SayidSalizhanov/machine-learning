from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main():
    irises = load_iris()
    data = irises.data
    target = irises.target

    pca = PCA(n_components=3)
    pca.fit(data)
    data_pca = pca.transform(data)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)
    predicted_labels = kmeans.predict(data)
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=predicted_labels)
    plt.show()

if __name__ == "__main__":
    main()