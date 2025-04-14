import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def main():
    data = pd.read_csv('C:\\Users\\sayid\\PycharmProjects\\MachineLearning\\resources\\bikes_rent.csv')
    # sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    # plt.show()

    data = data.drop(["season", "atemp", "windspeed(mph)"], axis=1)
    x,y = data.drop(["cnt"], axis=1), data["cnt"]

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    lr = LinearRegression()
    lr.fit(x_train,y_train)
    print(lr.score(x_test,y_test))

    predict = lr.predict(x_test)
    A = np.sum((y_test - predict) ** 2)
    B = np.sum((y_test - y_test) ** 2)

if __name__ == '__main__':
    main()