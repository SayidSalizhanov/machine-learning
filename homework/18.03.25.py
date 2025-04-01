import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from itertools import combinations


# Метод локтя
def find_optimal_clusters(data, max_k=10):
    inertia = []
    k_values = range(1, max_k + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    # Визуализация метода локтя
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia, marker="o")
    plt.title("Метод локтя для выбора оптимального числа кластеров")
    plt.xlabel("Количество кластеров (k)")
    plt.ylabel("Сумма квадратов расстояний (Inertia)")
    plt.show()


# Реализация k-means вручную
def kmeans_manual(data, k, max_iters=100):

    def initialize_centroids(data, k):
        # Инициализирует центроиды случайными точками из данных
        random_indices = np.random.choice(data.shape[0], k, replace=False)
        return data[random_indices]

    def assign_clusters(data, centroids):
        # Назначает кластеры для каждой точки на основе расстояний до центроидов
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(data, labels, k):
        # Обновляет координаты центроидов как среднее всех точек в каждом кластере
        return np.array([data[labels == i].mean(axis=0) for i in range(k)])

    def plot_iteration(data, centroids, labels, iteration):
        # Визуализирует текущее состояние кластеров и центроидов
        plt.figure(figsize=(6, 4))
        for cluster in range(k):
            cluster_points = data[np.array(labels) == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Кластер {cluster}")
        plt.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="x", s=100, label="Центроиды")
        plt.title(f"Итерация {iteration + 1}")
        plt.legend()
        plt.show()

    centroids = initialize_centroids(data, k)
    history = []  # История для визуализации

    for i in range(max_iters):
        labels = assign_clusters(data, centroids)
        history.append((centroids.copy(), labels))

        # Визуализация текущей итерации
        plot_iteration(data, centroids, labels, i)

        new_centroids = update_centroids(data, labels, k)

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return history, labels, centroids


# Визуализация кластеров на всех возможных проекциях
def plot_final_clusters(data, labels, num_clusters):
    feature_pairs = list(combinations(range(data.shape[1]), 2))
    plt.figure(figsize=(15, 10))

    for i, (f1, f2) in enumerate(feature_pairs):
        plt.subplot(3, 2, i + 1)
        for cluster in range(num_clusters):
            cluster_points = data[np.array(labels) == cluster]
            plt.scatter(cluster_points[:, f1], cluster_points[:, f2], label=f"Кластер {cluster}")
        plt.xlabel(f"Feature {f1}")
        plt.ylabel(f"Feature {f2}")
        plt.legend()
        plt.title(f"Проекция: признак {f1} и признак {f2}")

    plt.tight_layout()
    plt.show()


def main():
    # Загрузить данные ирисов
    iris = load_iris()
    X = iris['data']
    X_2d = X[:, :2]  # Для 2D-визуализации используем только первые два признака

    # Метод локтя
    print("Метод локтя")
    find_optimal_clusters(X)

    # Реализация k-means вручную
    print("\nРеализация k-means вручную")
    k = 3
    history, labels, centroids = kmeans_manual(X_2d, k)

    # Финальная визуализация кластеров
    print("\nФинальная визуализация кластеров")
    plot_final_clusters(X, labels, k)


if __name__ == "__main__":
    main()