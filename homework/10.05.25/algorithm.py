import pandas as pd
import numpy as np
import requests
from deap import base, creator, tools, algorithms
import random
import folium
import time

# Параметры задачи
FILENAME = "C:\\Users\\sayid\\PycharmProjects\\MachineLearning\\resources\\points.csv"
TRANSPORT = 'car'  # 'foot', 'car', 'bike'
TIME_LIMIT = 10     # Дни (можно менять)
POP_SIZE = 100      # Размер популяции
NGEN = 100          # Кол-во поколений

# ==== 1. Чтение точек ====
points = pd.read_csv(FILENAME)
n = len(points)

# ==== 2. Получаем матрицу времени ====
def get_travel_time(lat1, lon1, lat2, lon2, mode):
    url = f"http://router.project-osrm.org/route/v1/{mode}/{lon1},{lat1};{lon2},{lat2}?overview=false"
    try:
        r = requests.get(url, timeout=5)
        routes = r.json()
        if "routes" in routes and routes["routes"]:
            return routes["routes"][0]["duration"] / 60 / 60 / 24  # дни
    except Exception as ex:
        print(f"Ошибка API: {ex}")
    return 1e6  # очень большое время если ошибка (имитация невозможности пройти)

print("Строим матрицу времени, может занять до минуты...")
matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            matrix[i, j] = 0
        else:
            matrix[i, j] = get_travel_time(points.loc[i, "latitude"], points.loc[i, "longitude"],
                                           points.loc[j, "latitude"], points.loc[j, "longitude"],
                                           mode=TRANSPORT)
    time.sleep(0.2)  # чтоб не спамить OSRM

# ==== 3. Генетический алгоритм ====
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def evaluate(individual):
    total_time = 0
    total_priority = points.loc[individual[0], 'priority']
    for idx1, idx2 in zip(individual[:-1], individual[1:]):
        total_time += matrix[idx1, idx2]
        if total_time > TIME_LIMIT:
            return 0.,
        total_priority += points.loc[idx2, 'priority']
    return total_priority,

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(n), n)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

print("Запуск генетического алгоритма...")
population = toolbox.population(n=POP_SIZE)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("mean", np.mean)

result_pop, log = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2,
                                      ngen=NGEN, stats=stats, verbose=True)

# ==== 4. Извлекаем лучший маршрут ====
best = tools.selBest(result_pop, 1)[0]
route = [ (points.loc[idx, "latitude"], points.loc[idx, "longitude"]) for idx in best ]
route_names = [ points.loc[idx, "name"] for idx in best ]
route_prior = sum([points.loc[idx, "priority"] for idx in best])
route_time = 0
for idx1, idx2 in zip(best[:-1], best[1:]):
    route_time += matrix[idx1, idx2]

print("\nЛучший маршрут:")
for i, name in enumerate(route_names):
    print(f"{i+1}. {name} ({points.loc[best[i], 'priority']})")

print(f"Суммарный приоритет: {route_prior}")
print(f"Примерное время маршрута: {route_time:.1f} дней")

# ==== 5. Визуализация маршрута ====
m = folium.Map(location=route[0], zoom_start=12)
folium.Marker(route[0], popup="Старт: " + route_names[0],
              icon=folium.Icon(color='green')).add_to(m)
for latlon, name in zip(route[1:], route_names[1:]):
    folium.Marker(latlon, popup=name).add_to(m)
folium.PolyLine(route, color="red", weight=4).add_to(m)
m.save("best_route_kazan.html")
print("Карта маршрута сохранена как best_route_kazan.html")