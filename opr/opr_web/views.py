import base64
import io
import urllib.parse

from django.http import HttpResponse
from django.shortcuts import render
from django.template import context
from pulp import *
import time
from deap import base, algorithms, creator, tools

import random
import matplotlib.pyplot as plt
import numpy as np


# Create your views here.

def index(request):
    return render(request, "init.html")


init_params = {}


def init(request):
    init_params['percent'] = float(request.POST['percent'])
    init_params['percent2'] = float(request.POST['percent2'])
    init_params['additional'] = float(request.POST['additional'])
    init_params['x1_coef'] = int(request.POST['x1_coef']) * init_params['percent']
    init_params['x2_coef'] = int(request.POST['x2_coef']) * init_params['percent2']
    init_params['mat_sum'] = int(request.POST['mat_sum']) * init_params['additional']
    init_params['x1_mat'] = int(request.POST['x1_mat'])
    init_params['x2_mat'] = int(request.POST['x2_mat'])
    init_params['ogr2'] = float(request.POST['ogr2'])/100
    init_params['POPULATION_SIZE'] = int(request.POST['POPULATION_SIZE'])
    init_params['MAX_GENERATIONS'] = int(request.POST['MAX_GENERATIONS'])
    init_params['P_CROSSOVER'] = float(request.POST['P_CROSSOVER'])/100
    init_params['P_MUTATION'] = float(request.POST['P_MUTATION'])/100

    init_params['button'] = request.POST['button']

    if init_params['button'] == "Решить Симплекс методом":
        pr = Simpleks(init_params['x1_mat'],
                      init_params['x2_mat'],
                      init_params['mat_sum'],
                      init_params['ogr2'] - 1,
                      init_params['ogr2'],
                      0,
                      init_params['x1_coef'],
                      init_params['x2_coef'])
        pr.Calculate()
        type_count = 0
        if pr.simp_table[0][0] > 0:
            type_count += 1
        if pr.simp_table[1][0] > 0:
            type_count += 1
        x1 = pr.simp_table[0][0]
        x2 = pr.simp_table[1][0]
        x1p = init_params['x1_coef']
        x2p = init_params['x2_coef']
        x1mat = init_params['x1_mat']
        x2mat = init_params['x2_mat']
        if init_params['ogr2']==1:
            shadow_price_2=0
        else:
            shadow_price_2=(x2p * x2) / (x2mat * x2)
        return render(request, "result_simplex.html", {"x1": int(pr.simp_table[0][0]),
                                                       "x2": round(pr.simp_table[1][0]),
                                                       "sum": pr.simp_table[2][0],
                                                       "type_count": type_count,
                                                       "shadow_price_1": (x1p * x1) / (x1mat * x1),
                                                       "shadow_price_2": shadow_price_2,
                                                       "x1p": x1p, "x2p": x2p, "x1mat": x1mat, "x2mat": x2mat,
                                                       "mat_sum": init_params['mat_sum']

                                                       })

    elif init_params['button'] == "Решить Генетическим алгоритмом":
        # константы задачи
        ONE_MAX_LENGTH = 100  # длина подлежащей оптимизации битовой строки

        # константы генетического алгоритма
        POPULATION_SIZE = init_params['POPULATION_SIZE']  # количество индивидуумов в популяции
        P_CROSSOVER = init_params['P_CROSSOVER']  # вероятность скрещивания
        P_MUTATION = init_params['P_MUTATION']  # вероятность мутации индивидуума
        MAX_GENERATIONS = init_params['MAX_GENERATIONS']  # максимальное количество поколений

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        def evaluate(individual):
            rez = init_params['x1_coef'] * individual[0] + init_params['x2_coef'] * individual[1]
            return rez,  # кортеж

        def feasible(individual):
            """Проверка вхождения в ограничения результатов генетического алгоритма"""
            if init_params['ogr2'] * (individual[0] + individual[1]) <= individual[0] and \
                    init_params['x1_mat'] * individual[0] + init_params['x2_mat'] * individual[1] <= \
                    init_params['mat_sum']:
                return True
            return False

        toolbox = base.Toolbox()

        toolbox.register("zeroOrOne", random.randint, 0, 100)
        toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, 2)
        toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

        population = toolbox.populationCreator(n=POPULATION_SIZE)

        toolbox.register("evaluate", evaluate)
        toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 7.0, ))
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        stats.register("avg", np.mean)

        # define the hall-of-fame object:
        hof = tools.HallOfFame(MAX_GENERATIONS)  # по сути тут количество точек можно ставить больше


        population, logbook = algorithms.eaSimple(population, toolbox,
                                                  cxpb=P_CROSSOVER,
                                                  mutpb=P_MUTATION,
                                                  ngen=MAX_GENERATIONS,
                                                  stats=stats,
                                                  halloffame=hof,
                                                  verbose=True)
        best = hof.items[0]
        print("-- Best Ever Individual = ", best)
        print("-- Best Ever Fitness = ", best.fitness.values[0])

        maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

        hof_x = [x[0] for x in hof.items]
        hof_y = [x[1] for x in hof.items]
        fig, ax = plt.subplots(figsize=(10, 10)) # fisize - размер графика сечас 10 на 10
        ax.scatter(hof_x, hof_y, label='Максимальное значение в поколении')
        ax.scatter(best[0], best[1], label='Лучший результат')

        plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.grid() # сетка на графике
        plt.title('Максимальные результаты во всех поколениях')

        fig = plt.gcf()
        # convert graph into dtring buffer and then we convert 64 bit code into image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        type_count = 0
        if best[0] > 0:
            type_count += 1
        if best[1] > 0:
            type_count += 1
        x1 = best[0]
        x2 = best[1]
        x1p = init_params['x1_coef']
        x2p = init_params['x2_coef']
        x1mat = init_params['x1_mat']
        x2mat = init_params['x2_mat']
        if x2>0:
            shadow_price_2 = (x2p * x2) / (x2mat * x2)
        else:
            shadow_price_2 = 0
        return render(request, "result_genetic.html",
                      {"chart": uri, "x1": best[0], "x2": best[1], "sum": best.fitness.values[0],
                       "type_count": type_count,
                       "shadow_price_1": (x1p * x1) / (x1mat * x1),
                       "shadow_price_2": shadow_price_2,
                       "x1p": x1p, "x2p": x2p, "x1mat": x1mat, "x2mat": x2mat,
                       "mat_sum": init_params['mat_sum']
                       })


class Simpleks:
    def __init__(self, price1, price2, cost, req_area1, req_area2, area, out_put1, out_put2):
        self.simp_table = [[cost, price1, price2, 1, 0],
                           [area, req_area1, req_area2, 0, 1],
                           [0, -out_put1, -out_put2, 0, 0]]
        for i in range(len(self.simp_table)):
            for j in range(len(self.simp_table[i])):
                print((self.simp_table[i][j]), end=" ")
            print()
        self.basic_row = ["x3", "x4"]
        self.basic_col = ["x1", "x2", "x3", "x4"]

    def findMainCol(self):
        mainCol = 1
        m = len(self.simp_table)
        n = len(self.simp_table[m - 1])
        for j in range(2, n):
            if self.simp_table[m - 1][j] < self.simp_table[m - 1][mainCol]:
                mainCol = j
        print(mainCol)
        return mainCol

    def findMainRow(self, maincol):
        mainrow = 0
        m = len(self.simp_table)
        n = len(self.simp_table[m - 1])
        for i in range(0, m - 1):
            if self.simp_table[i][maincol] > 0:
                mainrow = i
                break
        for i in range(mainrow + 1, m - 1):
            if ((self.simp_table[i][maincol] > 0) and
                    ((self.simp_table[i][0] / self.simp_table[i][maincol]) < (
                            self.simp_table[mainrow][0] / self.simp_table[mainrow][maincol]))):
                mainrow = i
        self.basic_row[mainrow] = self.basic_col[maincol - 1]
        return mainrow

    def IsItEnd(self):
        flag = False
        m = len(self.simp_table)
        n = len(self.simp_table[m - 1])
        for i in range(1, n):
            if self.simp_table[m - 1][i] < 0:
                flag = True
                break
        return flag

    def Calculate(self):
        count = 1
        while self.IsItEnd():
            maincol = self.findMainCol()
            mainrow = self.findMainRow(maincol)
            m = len(self.simp_table)
            n = len(self.simp_table[m - 1])
            new_table = self.simp_table

            for j in range(0, n):
                new_table[mainrow][j] = self.simp_table[mainrow][j] / self.simp_table[mainrow][maincol]

            for i in range(0, m):
                if i == mainrow:
                    continue
                for j in range(0, n):
                    new_table[i][j] = self.simp_table[i][j] - self.simp_table[i][maincol] * self.simp_table[mainrow][j]
            print(count, "этап решения:\n")
            for i in range(len(self.simp_table)):
                for j in range(len(self.simp_table[i])):
                    print((self.simp_table[i][j]), end=" ")
                print()
            count = count + 1
            self.simp_table = new_table
        print("Конечная таблица:\n")
        x1 = LpVariable("x1", lowBound=0, cat='Integer')
        x2 = LpVariable("x2", lowBound=0, cat='Integer')
        problem = LpProblem('0', LpMaximize)
        problem += init_params['x1_coef'] * x1 + init_params['x2_coef'] * x2, "Функция цели"
        problem += init_params['ogr2'] * (x1 + x2) <= x1, "1"
        problem += init_params['x1_mat'] * x1 + init_params['x2_mat'] * x2 <= init_params['mat_sum'], "2"
        problem.solve()
        res = {}

        for variable in problem.variables():
            res[variable.name] = variable.varValue
        self.simp_table[0][0] = res['x1']
        self.simp_table[0][1] = res['x2']
        for i in range(len(self.simp_table)):
            for j in range(len(self.simp_table[i])):
                print((self.simp_table[i][j]), end=" ")
            print()
        print(self.basic_row[0], "=", self.simp_table[0][0], "\n")
        print(self.basic_row[1], "=", self.simp_table[1][0], "\n")
