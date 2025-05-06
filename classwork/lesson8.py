import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv("C:\\Users\\sayid\\PycharmProjects\\MachineLearning\\resources\\AmesHousing.csv")
    data = data[["Lot Area", "Lot Frontage", "SalePrice"]]

    data = data.fillna(data.mean(), inplace=True)
    X, y = data[["Lot Area", "Lot Frontage"]], data["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    Dtree = DecisionTreeRegressor()
    Dtree.fit(X_train, y_train)
    print(Dtree.score(X_test, y_test))

    plt.figure(figsize=(20, 10))
    plot_tree(Dtree, feature_names=X.columns, filled=True)
    plt.show()