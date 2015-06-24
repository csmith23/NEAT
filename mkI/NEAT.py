__author__ = 'Coleman'

import random

class Species:
    def __init__(self):
        self.representative = None
        self.organisms = []
        self.topFitness = 0
        self.averageFitness = 0
        self.staleness = 0

    def inSpecies(self, genome):
        pass

    def adjustedFitness(self, genome):
        pass

    def calculateFitness(self):
        pass

    def cull(self, newGeneration):
        pass

    def breedChild(self):
        pass

    def calculateStaleness(self):
        pass

class Organism:
    def __init__(self, inputs, outputs):
        if genome is None:
            self.genome = Genome(inputs, outputs)

        else:
            self.genome = genome

        self.network = Network(self.genome)
        self.fitness = 0
        self.adjustedFitness = 0

    def breed(self, organism):
        pass

    def calculateFitness(self):
        pass

class Genome:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.genes = []
        self.numNeurons = 0

    def copy(self):
        newGenome = Genome(self.inputs, self.outputs)
        newGenome.genes = [gene.copy() for gene in self.genes]
        newGenome.numNeurons = self.numNeurons

        return newGenome

    def sortGenes(self):
        pass

    def disjoint(self, genome):
        pass

    def excess(self, genome):
        pass

    def weight(self, genome):
        pass

    def distance(self, genome):
        pass

    def mutate(self):
        pass

    def pointMutate(self, perturbRate):
        pass

    def linkMutate(self):
        pass

    def nodeMutate(self):
        pass

    def enableMutate(self):
        pass

    def disableMutate(self):
        pass

    def crossover(self, genome):
        pass

class Gene:
    def __init__(self, innovation, input, output):
        self.input = input
        self.output = output
        self.weight = random.random()
        self.innovation = innovation
        self.enabled = True
        self.recurrent = False

    def copy(self):
        newGene = Gene(self.innovation, self.input, self.output)
        newGene.weight = self.weight
        newGene.enabled = self.enabled

        return newGene

class Network:
    def __init__(self, genome):
        global recurrent
        self.neurons = {}
        self.outputs = {}
        for input in genome.inputs:
            self.neurons[input] = Neuron(0)

        genome.sortGenes()
        for gene in genome.genes:
            if gene.enabled:
                if gene.input not in list(self.neurons.keys()):
                    self.neurons[gene.input] = Neuron(0)

                if gene.output not in list(self.neurons.keys()):
                    self.neurons[gene.output] = Neuron(self.neurons[gene.input].timeStep + 1)

                if gene.input.timeStep > gene.output.timeStep and recurrent:
                    gene.recurrent = True
                    self.neurons[gene.output].inputs.append(gene)

                elif gene.input.timeStep == gene.output.timeStep:
                    gene.output.timeStep += 1
                    self.neurons[gene.output].inputs.append(gene)

        for output in genome.outputs:
            found = False
            for key in list(self.neurons.keys()):
                if key is output:
                    found = True

            if not found:
                self.neurons[output] = Neuron(0)

            self.outputs[output] = self.neurons[output]

    def eval(self, inputs):
        for name in list(inputs.keys()):
            self.neurons[name] = inputs[name]

        for key in (self.neurons.keys()):
            neuron = self.neurons[key]
            neuron.sigmoid()

        outputs = {}
        for key in (self.outputs.keys()):
            outputs[key] = self.outputs[key].value

        return outputs

class Neuron:
    def __init__(self, timeStep):
        self.value = 0
        self.lastValue = 0
        self.inputs = []
        self.timeStep = timeStep

    def sigmoid(self):
        global steepness
        self.lastValue = self.value
        values = []
        for gene in self.inputs:
            if gene.recurrent:
                values.append(gene.input.lastValue)

            else:
                values.append(gene.input.value)

        x = sum(values)
        self.value = 2 / (steepness ** x + 1) - 1
        self.inputs = []

currentInnovation = -1
def newInnovation():
    global currentInnovation
    currentInnovation += 1
    return currentInnovation

steepness = 5
distanceThreshold = 0
recurrent = False

population = []
species = []

def rankGlobally(genome):
    pass