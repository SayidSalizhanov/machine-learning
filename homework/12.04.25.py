import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Загрузка данных
df = pd.read_csv('C:\\Users\\sayid\\PycharmProjects\\MachineLearning\\resources\\AmesHousing.csv')

# Разделение на числовые и категориальные признаки
numeric_features = df.select_dtypes(include=[np.number]).columns
categorical_features = df.select_dtypes(exclude=[np.number]).columns

# Обработка категориальных признаков: one-hot encoding
df_encoded = pd.get_dummies(df[categorical_features], drop_first=True)

# Объединение числовых признаков с обработанными категориальными
X = pd.concat([df[numeric_features], df_encoded], axis=1)
y = df['SalePrice']

# Обработка пропущенных значений
X = X.fillna(X.mean(numeric_only=True))

# Удаление коррелирующих числовых признаков
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X = X.drop(to_drop, axis=1)

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Уменьшение размерности для 3D графика
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Построение 3D графика
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], y, c=y, cmap='viridis', s=5)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('SalePrice')
plt.title('3D Visualization of Features and Target')
plt.show()

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Подбор параметра alpha для Lasso
alphas = np.logspace(-4, 1, 50)
rmse_values = []

for alpha in alphas:
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_values.append(rmse)

# Построение графика RMSE vs Alpha
plt.figure(figsize=(10, 6))
plt.plot(alphas, rmse_values, marker='o')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.title('RMSE vs Alpha for Lasso Regression')
plt.grid(True)
plt.show()

# Определение важного признака
best_alpha = alphas[np.argmin(rmse_values)]
model = Lasso(alpha=best_alpha)
model.fit(X_train, y_train)
coefficients = model.coef_
feature_names = X.columns

# Находим признак с наибольшим абсолютным коэффициентом
max_coeff_index = np.argmax(np.abs(coefficients))
most_influential = feature_names[max_coeff_index]

print(f"Наибольшее влияние оказывает признак: {most_influential}")
print(f"Оптимальный RMSE при alpha={best_alpha:.4f}: {min(rmse_values):.2f}")