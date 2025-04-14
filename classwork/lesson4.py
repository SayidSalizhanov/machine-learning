import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

def main():
    data = []
    for i in range(3):
        centerX = random.random() * 5
        centerY = random.random() * 5
        for j in range(30):
            data.append([random.gauss(centerX, 0.5), random.gauss(centerY, 0.5)])
    data = np.array(data)

    dbscan = DBSCAN(eps=0.5, min_samples=3)
    dbscan.fit(data)
    predict = dbscan.labels_
    plt.scatter(data[:, 0], data[:, 1], c=predict)
    plt.show()

if __name__ == '__main__':
    main()