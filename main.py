from deap import base
from deap import creator
from deap import tools
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import algelitism
import time

# 0.01 - start, 0.15 - quit

labirint = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 9.01,
            1.01, 9.02, 1.02, 9.03, 1.04, 9.04, 9.05, 0.09, 9.06,
            1.05, 1.06, 1.03, 1.08, 1.09, 1.66, 1.10, 0.10, 9.08,
            9.09, 9.10, 1.11, 9.11, 9.12, 1.12, 1.63, 0.11, 9.14,
            1.13, 9.15, 1.88, 1.15, 1.16, 1.17, 1.18, 0.12, 9.16,
            1.19, 9.17, 9.18, 9.19, 9.20, 1.20, 9.21, 0.13, 9.22,
            1.21, 9.23, 1.22, 9.24, 1.23, 1.24, 1.25, 0.14, 9.25,
            1.26, 1.27, 1.28, 1.29, 1.30, 1.31, 9.26, 0.15, 9.26]

hod = 14
POPULATION_SIZE = 50
P_CROSSOVER = 0.2
P_MUTATION = 0.9
MAX_GENERATIONS = 50

HALL_OF_FAME_SIZE = 1
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def CreateIndividual():
    choose = [1, 2]
    compas = 0
    ind = []
    for i in range(hod):
        if random.choice(choose) == 1:
            if compas in [8, 17, 26, 35, 44, 53, 62]:
                compas += 9
            else:
                compas += 1
            ind.append(labirint[compas])
        else:
            if compas >= 63:
                compas += 1
            else:
                compas += 9
            ind.append(labirint[compas])
    return creator.Individual(ind)


def IndividualFitness(individual):
    return sum(individual),


def mutateIndividual(individual, indpb):
    if random.random() < indpb:
        slt = random.choice([5, 6, 7])
        p = individual[:slt]
        choose = [1, 2]
        compas = labirint.index(p[-1])
        new = []
        for i in range(hod-slt):
            if random.choice(choose) == 1:
                if compas in [8, 17, 26, 35, 44, 53, 62]:
                    compas += 9
                else:
                    compas += 1
                new.append(labirint[compas])
            else:
                if compas >= 63:
                    compas += 1
                else:
                    compas += 9
                new.append(labirint[compas])
        ind = p[:slt]+new
        for i in range(len(ind)):
            individual[i] = ind[i]
        return individual,


def show(ind):
    print(ind)
    print(' ')
    ax.clear()
    vertex = []
    y = 20
    for j in range(8):
        for i in range(9):
            vertex.append((i, y))
        y -= 2
    vx = [v[0] for v in vertex]
    vy = [v[1] for v in vertex]
    color = []
    for i in labirint:
        if 0 < i < 9:
            color.append('gray')
        else:
            color.append('red')
    for i in range(72):
        ax.plot(vx[i], vy[i], ' ob', markersize=10, color=color[i])
    xy = []
    for i in range(len(ind)):
        xy.append(vertex[labirint.index(ind[i])])
        if color[labirint.index(ind[i])] != 'red':
            color[labirint.index(ind[i])] = 'green'
    for i in range(72):
        ax.plot(vx[i], vy[i], ' ob', markersize=10, color=color[i])
    ax.add_line(Line2D((0, xy[0][0]), (20, xy[0][1]), color='#aaa'))
    ax.plot(0, 20, ' ob', markersize=10, color='yellow')
    ax.plot(7, 6, ' ob', markersize=10, color='blue')
    for i in range(len(xy)):
        if i == 13:
            pass
        else:
            ax.add_line(Line2D((xy[i][0], xy[i+1][0]), (xy[i][1], xy[i+1][1]), color='#aaa'))
    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(0.2)


plt.ion()
fig, ax = plt.subplots()

toolbox = base.Toolbox()
toolbox.register("populationCreator", tools.initRepeat, list, CreateIndividual)
population = toolbox.populationCreator(n=POPULATION_SIZE)
toolbox.register("evaluate", IndividualFitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", mutateIndividual, indpb=10)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)

population, logbook = algelitism.eaSimpleElitism(population, toolbox,
                                        cxpb=P_CROSSOVER,
                                        mutpb=P_MUTATION,
                                        ngen=MAX_GENERATIONS,
                                        halloffame=hof,
                                        stats=stats,
                                        callback=(show, (hof)),
                                        verbose=True)
plt.ioff()
plt.show()

print('\nPerson found exit: '+str(hof.items[0]))

maxFitnessValues, meanFitnessValues = logbook.select("min", "avg")
plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()
