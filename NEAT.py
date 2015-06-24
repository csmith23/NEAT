__author__ = 'Coleman'

import random

class Population:
    def __init__(self, size):
        self.species = [Species()]
        self.species[0].representative = Organism()
        self.species[0].organisms = self.organisms

        self.size = size
        self.innovation = 0
        self.innovationGenes = []
        self.organisms = [Organism() for i in range(size)]
        self.IO = []
        self.epoch = 0
        self.parameter = {"stale": 0,
                          "threshold": 0,
                          "c1": 0,
                          "c2": 0,
                          "c3": 0,
                          "perturbLimit": 0,
                          "bias": 0,
                          "perturb": 0,
                          "weight": 0,
                          "node": 0,
                          "link": 0,
                          "disable": 0,
                          "enable": 0,
                          "crossover": 0}

    def rankGlobally(self):
        pass

    def newInnovation(self, gene):
        pass

class Species:
    def __init__(self):
        self.representative = None
        self.organisms = []
        self.topFitness = 0
        self.averageFitness = 0

    def inSpecies(self, organism):
        pass

    def adjustFitness(self):
        pass

    def cull(self, newGeneration=False):
        pass

    def breed(self):
        pass

class Organism:
    def __init__(self):
        self.genome = Genome()
        self.network = Network(self.genome)
        self.fitness = 0
        self.adjustedFitness = 0
        self.age = 0
        self.elite = False

    def breed(self, other):
        pass

    def calculateFitness(self):
        # implemented by experiment organisms
        pass

class Genome:
    def __init__(self):
        self.population = None
        self.gene = []
        self.numNeurons = 0

    def distance(self, other):
        pass

    def disjoint(self, other):
        pass

    def excess(self, other):
        pass

    def weight(self, other):
        pass

    def mutate(self):
        pass

    def point(self):
        pass

    def link(self):
        pass

    def node(self):
        pass

    def enable(self):
        pass

    def disable(self):
        pass

    def contains(self, gene):
        pass

    def copy(self):
        pass

    def crossover(self, other):
        pass

class Gene:
    def __init__(self, input, output, innovation):
        self.input = input
        self.output = output
        self.weight = random.uniform(-1, 1)
        self.enabled = True
        self.recurrent = False
        self.innovation = innovation
        self.age = 0

    def copy(self):
        pass

class Network:
    def __init__(self, genome):
        self.neurons = {}
        pass

    def eval(self, inputs):
        pass

class Neuron:
    def __init__(self):
        self.value = 0
        self.lastValue = 0
        self.sources = []
        self.timeStep = 0

    def sigmoid(self):
        pass
