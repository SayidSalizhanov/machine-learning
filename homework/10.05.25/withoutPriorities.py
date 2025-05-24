import pandas as pd
import numpy as np
import requests
from deap import base, creator, tools, algorithms
import random
import folium
import time

# Параметры
FILENAME = "C:\\Users\\sayid\\PycharmProjects\\MachineLearning\\resources\\points2.csv"
TRANSPORT = 'car'
POP_SIZE = 100
NGEN = 100

# 1. Загрузка точек
points = pd.read_csv(FILENAME)
n = len(points)

# 2. Получение матрицы времени
def get_travel_time(lat1, lon1, lat2, lon2, mode):
    url = f"http://router.project-osrm.org/route/v1/{mode}/{lon1},{lat1};{lon2},{lat2}?overview=false"
    try:
        r = requests.get(url, timeout=5)
        routes = r.json()
        if "routes" in routes and routes["routes"]:
            return routes["routes"][0]["duration"] / 60  # минуты
    except Exception as ex:
        print(f"Ошибка API: {ex}")
    return 1e6  # большое значение при ошибке

print("Строим матрицу времени...")
matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            matrix[i, j] = 0
        else:
            matrix[i, j] = get_travel_time(
                points.loc[i, "latitude"], points.loc[i, "longitude"],
                points.loc[j, "latitude"], points.loc[j, "longitude"],
                mode=TRANSPORT
            )
    time.sleep(0.2)

# 3. Генетический алгоритм (без приоритетов)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def evaluate(individual):
    total_time = 0
    for idx1, idx2 in zip(individual[:-1], individual[1:]):
        total_time += matrix[idx1, idx2]
    return total_time,

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(n), n)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

print("Запуск алгоритма оптимизации...")
population = toolbox.population(n=POP_SIZE)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("mean", np.mean)

result_pop, log = algorithms.eaSimple(
    population, toolbox,
    cxpb=0.7, mutpb=0.2,
    ngen=NGEN, stats=stats, verbose=True
)

# 4. Результат
best = tools.selBest(result_pop, 1)[0]
route = [ (points.loc[idx, "latitude"], points.loc[idx, "longitude"]) for idx in best ]
route_names = [ points.loc[idx, "name"] for idx in best ]
route_time = sum(matrix[i, j] for i, j in zip(best[:-1], best[1:]))

print("\nОптимальный маршрут:")
for i, name in enumerate(route_names):
    print(f"{i+1}. {name}")
print(f"Суммарное время маршрута: {route_time:.1f} мин")

# 5. Визуализация маршрута по дорогам
m = folium.Map(location=route[0], zoom_start=12)

# Добавляем первую точку
folium.Marker(route[0], popup="Старт: " + route_names[0],
              icon=folium.Icon(color='green')).add_to(m)

# Добавляем остальные точки
for latlon, name in zip(route[1:], route_names[1:]):
    folium.Marker(latlon, popup=name).add_to(m)

# Рисуем маршрут по дорогам
for i in range(len(route) - 1):
    lat1, lon1 = route[i]
    lat2, lon2 = route[i + 1]
    url = f"http://router.project-osrm.org/route/v1/{TRANSPORT}/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"

    try:
        r = requests.get(url)
        data = r.json()
        if "routes" in data and data["routes"]:
            geometry = data["routes"][0]["geometry"]
            folium.GeoJson(geometry, name=f"Route {i}").add_to(m)
        else:
            print(f"Не удалось получить маршрут между точками {i} и {i+1}")
    except Exception as e:
        print(f"Ошибка запроса маршрута {i} → {i+1}: {e}")
    time.sleep(0.1)

m.save("shortest_route_kazan.html")
print("Карта маршрута сохранена как shortest_route_kazan.html")

