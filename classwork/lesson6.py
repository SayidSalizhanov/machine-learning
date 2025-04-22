import pandas as pd
import numpy as np

# Загрузка данных
disease_data = pd.read_csv('C:\\Users\\sayid\\PycharmProjects\\MachineLearning\\resources\\disease.csv', delimiter=';')
symptom_data = pd.read_csv('C:\\Users\\sayid\\PycharmProjects\\MachineLearning\\resources\\symptom.csv', delimiter=';')

# Сопоставление названий болезней и столбцов
disease_columns = [
    'Острый левостор паратонз абсцесс',
    'Острый правостор паратонз абсцесс',
    'Острый двухстор паратонз абсцесс',
    'Острый левостор паратон-зиллит',
    'Острый правостор паратон-зиллит',
    'Острый левостор парафарин абсцесс',
    'Острый правостор парафарин абсцесс',
    'Острый левостор парафарингит',
    'Острый правостор парафарингит'
]
disease_names = [
    'Левосторонний паратонзиллярный абсцесс',
    'Правосторонний паратонзиллярный абсцесс',
    'Двусторонний паратонзиллярный абсцесс',
    'Левосторонний паратонзиллит',
    'Правосторонний паратонзиллит',
    'Левосторнний парафарингиальный абсцесс',
    'Правосторонний парафарингиальный абсцесс',
    'Левосторонний парафарингит',
    'Правосторонний парафарингит'
]

# Получаем априорные вероятности
disease_counts = disease_data.set_index('disease').loc[disease_names]['количество пациентов'].values
priors = disease_counts / disease_counts.sum()

def bayesian_diagnosis(symptoms, symptom_data, disease_columns, priors):
    # symptoms — словарь: {название симптома: True/False}
    posteriors = []
    for i, disease_col in enumerate(disease_columns):
        likelihood = 1.0
        for symptom, present in symptoms.items():
            prob = symptom_data.loc[symptom_data['symptom'] == symptom, disease_col].values[0]
            if present:
                likelihood *= prob
            else:
                likelihood *= (1 - prob)
        posteriors.append(priors[i] * likelihood)
    posteriors = np.array(posteriors)
    posteriors /= posteriors.sum()
    return disease_names[np.argmax(posteriors)], posteriors

# Пример: пользователь вводит симптомы
symptoms_input = {
    'Недомогание': True,
    'Слабость': True,
    'Головную боль': False,
    'Отсутствие аппетита': True,
    'Болезненное открывание рта': True
}

diagnosis, probabilities = bayesian_diagnosis(symptoms_input, symptom_data, disease_columns, priors)
print("Наиболее вероятная болезнь:", diagnosis)
for name, prob in zip(disease_names, probabilities):
    print(f"{name}: {prob:.3f}")
