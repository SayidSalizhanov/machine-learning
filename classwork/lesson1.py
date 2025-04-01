import pandas as pd
import matplotlib.pyplot as plt

def main():
    dataset = pd.read_csv("../resources/train.csv")
    #dataset.groupby(["Sex"])["Survived"].value_counts().plot(kind="bar")
    plt.plot(range(0, 2), range(0, 2))
    plt.show()

main()