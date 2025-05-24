import pandas as pd
import numpy as np
import requests
from deap import base, creator, tools, algorithms
import random
import folium
import time

# Параметры задачи
FILENAME = "C:\\Users\\sayid\\PycharmProjects\\MachineLearning\\resources\\points2.csv"
TRANSPORT = 'car'  # 'foot', 'car', 'bike'
TIME_LIMIT = 5     # Дни (можно менять)
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

# ==== 4. Извлекаем лучший маршрут с учетом времени ====
best = tools.selBest(result_pop, 1)[0]

# Определяем точки, которые успевают пройти до превышения TIME_LIMIT
valid_route = []
current_time = 0.0
total_priority = 0

for i in range(len(best) - 1):
    from_idx = best[i]
    to_idx = best[i + 1]
    segment_time = matrix[from_idx, to_idx]

    if current_time + segment_time > TIME_LIMIT:
        break
    current_time += segment_time

    # Добавляем точку from_idx только при первом переходе
    if i == 0:
        valid_route.append(from_idx)
        total_priority += points.loc[from_idx, 'priority']

    valid_route.append(to_idx)
    total_priority += points.loc[to_idx, 'priority']

# Обрабатываем случай, когда даже первая точка не добавлена
if not valid_route:
    valid_route = [best[0]]
    total_priority = points.loc[best[0], 'priority']

route = [(points.loc[idx, "latitude"], points.loc[idx, "longitude"]) for idx in valid_route]
route_names = [points.loc[idx, "name"] for idx in valid_route]
route_time = current_time

print("\nЛучший маршрут (с учетом лимита времени):")
for i, name in enumerate(route_names):
    print(f"{i + 1}. {name} ({points.loc[valid_route[i], 'priority']})")

print(f"Суммарный приоритет: {total_priority}")
print(f"Примерное время маршрута: {route_time:.1f} дней")

# ==== 5. Визуализация маршрута по дорогам ====
m = folium.Map(location=route[0], zoom_start=12)

# Стартовая точка
folium.Marker(route[0], popup="Старт: " + route_names[0],
              icon=folium.Icon(color='green')).add_to(m)

# Промежуточные точки
for latlon, name in zip(route[1:-1], route_names[1:-1]):
    folium.Marker(latlon, popup=name).add_to(m)

# Финишная точка
if len(route) > 1:
    folium.Marker(route[-1], popup="Финиш: " + route_names[-1],
                  icon=folium.Icon(color='red')).add_to(m)

# Строим маршрут по дорогам между точками
for i in range(len(route) - 1):
    lat1, lon1 = route[i]
    lat2, lon2 = route[i + 1]
    url = f"http://router.project-osrm.org/route/v1/{TRANSPORT}/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    try:
        r = requests.get(url)
        data = r.json()
        if "routes" in data and data["routes"]:
            geometry = data["routes"][0]["geometry"]
            folium.GeoJson(geometry, name=f"Segment {i}").add_to(m)
        else:
            print(f"Не удалось получить маршрут {i} → {i+1}")
    except Exception as e:
        print(f"Ошибка запроса маршрута {i} → {i+1}: {e}")
    time.sleep(0.1)

m.save("best_route_time_limit_check_kazan.html")
print("Карта маршрута сохранена как best_route_time_limit_check_kazan.html")
