__author__ = 'Coleman'

import random

class Population:
    def __init__(self, size):
        self.parameters = {"inputs": ["A", "B"],
                           "outputs": ["OUT"],
                           "recurrent": False,
                           "steepness": 4.9,
                           "c1": 0,
                           "c2": 0,
                           "c3": 0,
                           "perturbLimit": 0,
                           "weightChange": 0,
                           "bias": 0,
                           "perturb": 0,
                           "weight": 0,
                           "node": 0,
                           "link": 0,
                           "disable": 0,
                           "enable": 0,
                           "crossover": 0}

        self.organisms = [Organism(self) for i in range(size)]
        self.species = [Species()]
        self.species[0].representative = Organism(self)
        self.species[0].organisms = self.organisms

        self.innovation = 0
        self.innovationGenes = []
        self.node = 0
        self.nodeGenes = []

        self.size = size
        self.threshold = 0
        self.deathAge = 0
        self.deathCount = 0
        self.cullLimit = 0

    def breedChild(self):
        self.rankGlobally()
        parent = None
        while True:
            parent = random.choice(self.organisms)
            if parent.elite:
                break

        otherParent = None
        while True:
            otherParent = random.choice(self.organisms)
            if otherParent.elite:
                break

        self.organisms.append(parent.breed(otherParent))
        for species in self.species:
            if species.addToSpecies(self.organisms[-1]):
                break

    def step(self):
        for organism in self.organisms:
            organism.age += 1
            if organism.age >= self.deathAge:
                organism.die()

        if self.deathCount > self.cullLimit:
            for species in self.species:
                species.cull()

        for organism in self.organisms:
            for species in self.species:
                if species.addToSpecies(organism):
                    break

    def rankGlobally(self):
        order = [self.organisms[0]]
        for organism in self.organisms[1:]:
            for i in range(len(order)):
                if organism.adjustedFitness > order[i].adjustedFitness:
                    order.insert(i, organism)
                    break

                if i == len(order) - 1:
                    order.append(organism)

        for organism in order[:len(order) // 2]:
            organism.elite = True

        for organism in order[len(order) // 2:]:
            organism.elite = False

    def getInnovation(self, input, output):
        for innovation in self.innovationGenes:
            innovationGene = innovation[0]
            if input == innovationGene[0] and output == innovationGene[1]:
                return innovation[1]

        self.innovation += 1
        self.innovationGenes.append(((input, output), self.innovation))

        return self.innovation

    def getNode(self, input, output):
        for node in self.nodeGenes:
            gene = node[0]
            if gene[0] == input and gene[1] == output:
                return node[0]

        self.node += 1
        self.nodeGenes.append(((input, output), self.node))

        return self.node

class Species:
    def __init__(self):
        self.representative = None
        self.organisms = []
        self.topFitness = 0
        self.averageFitness = 0

    def addToSpecies(self, organism):
        if self.representative.genome.distance(organism.genome) > self.representative.threshold:
            return False

        organism.species = self
        self.organisms.append(organism)

        return True

    def adjustFitness(self):
        fitness = []
        for organism in self.organisms:
            fitness.append(organism.fitness)

        fitnessSum = sum(fitness)
        for organism in self.organisms:
            organism.adjustedFitness = organism.fitness / fitnessSum

    def cull(self):
        newRepresentative = self.organisms[0]
        for organism in self.organisms:
            if organism.fitness > newRepresentative.fitness:
                newRepresentative = organism

        self.organisms = []

class Organism:
    def __init__(self, population, genome=None):
        self.population = population
        self.genome = genome
        if self.genome is None:
            self.genome = Genome(self, self.population.parameters)

        self.network = Network(self.genome)
        self.species = None
        self.fitness = 0
        self.adjustedFitness = 0
        self.age = 0
        self.elite = False

    def breed(self, other):
        newOrganism = Organism(self.population, genome=self.genome.crossover(other.genome))
        newOrganism.genome.organism = newOrganism

        return newOrganism

    def die(self):
        self.population.organisms.remove(self)
        self.species.organisms.remove(self)
        population.deathCount += 1
        population.breedChild()

    def mutate(self):
        self.genome.mutate()
        self.network = Network(self.genome)

    def calculateFitness(self):
        # implemented by experiment organisms, a subclass
        pass

class Genome:
    def __init__(self, organism, parameters):
        self.organism = organism
        self.parameters = parameters
        self.genes = []

    def distance(self, other):
        self.sortGenes()
        other.sortGenes()
        number = max(len(self.genes), len(other.genes))
        return ((self.parameters["c1"] * self.excess(other)) / number) + ((self.parameters["c2"] * self.disjoint(other)) / number) + (self.parameters["c3"] * self.weight(other))

    def disjoint(self, other):
        geneIndex = 0
        otherIndex = 0
        disjoint = 0
        while True:
            if geneIndex == len(self.genes) or otherIndex == len(other.genes):
                    break

            if self.genes[geneIndex].innovation == other.genes[otherIndex].innovation:
                geneIndex += 1
                otherIndex += 1

            elif self.genes[geneIndex].innovation < other.genes[otherIndex].innovation:
                geneIndex += 1
                disjoint += 1

            else:
                otherIndex += 1
                disjoint += 1

        return disjoint

    def excess(self, other):
        excessLimit = min(self.genes[-1].innovation, other.genes[-1].innovation)
        maxIndex = max(self.genes[-1].innovation, other.genes[-1].innovation)

        return maxIndex - excessLimit

    def weight(self, other):
        geneIndex = 0
        otherIndex = 0
        weights = []
        while True:
            if geneIndex == len(self.genes) or otherIndex == len(other.genes):
                    break

            if self.genes[geneIndex].innovation == other.genes[otherIndex].innovation:
                geneIndex += 1
                otherIndex += 1

                weights.append((self.genes[geneIndex] + other.genes[otherIndex]) / 2)

            else:
                if self.genes[geneIndex].innovation < other.genes[otherIndex].innovation:
                    geneIndex += 1

                else:
                    otherIndex += 1

        return sum(weights) / len(weights)

    def mutate(self):
        if random.random() < self.parameters["bias"]:
            self.link(True)

        if random.random() < self.parameters["weight"]:
            self.point()

        if random.random() < self.parameters["node"]:
            self.node()

        if random.random() < self.parameters["link"]:
            self.link()

        if random.random() < self.parameters["disable"]:
            self.disable()

        if random.random() < self.parameters["enable"]:
            self.enable()

    def point(self):
        if len(self.genes) == 0:
            return

        gene = random.choice(self.genes)
        newWeight = random.randint(-self.parameters["weightChange"] * gene.weight, self.parameters["weightChange"] * gene.weight)
        if random.random() < self.parameters["perturb"]:
            gene.weight += random.randint(-1, 1) * self.parameters["perturbLimit"]

        else:
            gene.weight = newWeight

        gene.age += 1

    def link(self, forceBias=False):
        input = self.randomNeuron(True)
        if forceBias:
            input = "bias"

        output = self.randomNeuron(False)
        gene = Gene(input, output, self.organism.population.getInnovation(input, output))
        if not self.contains(gene):
            self.genes.append(gene)

    def node(self):
        if len(self.genes) == 0:
            return

        gene = random.choice(self.genes)
        if not gene.enabled:
            return

        gene.enabled = False
        inputGene = Gene(gene.input, self.organism.population.getNode(gene.input, gene.output), self.organism.population.getInnovation(gene.input, self.organism.population.getNode(gene.input, gene.output)))
        outputGene = Gene(self.organism.population.getNode(gene.input, gene.output), gene.output, self.organism.population.getInnovation(self.organism.population.getNode(gene.input, gene.output), gene.output))
        inputGene.weight = 1
        outputGene.weight = gene.weight
        self.genes.append(inputGene)
        self.genes.append(outputGene)

    def enable(self):
        if len(self.genes) == 0:
            return

        gene = random.choice(self.genes)
        gene.enabled = True

    def disable(self):
        if len(self.genes) == 0:
            return

        gene = random.choice(self.genes)
        gene.enabled = False

    def contains(self, otherGene):
        for gene in self.genes:
            if gene.input == otherGene.input and gene.output == otherGene.output:
                return True

        return False

    def copy(self):
        newGenome = Genome(None, self.parameters)
        newGenome.genes = [gene.copy() for gene in self.genes]

        return newGenome

    def crossover(self, other):
        if random.random() < self.parameters["crossover"]:
            self.sortGenes()
            other.sortGenes()
            newGenome = Genome(None, self.parameters)
            geneIndex = 0
            otherIndex = 0
            while True:
                if geneIndex == len(self.genes) or otherIndex == len(other.genes):
                    break

                if self.genes[geneIndex].innovation == other.genes[otherIndex].innovation:
                    geneIndex += 1
                    otherIndex += 1

                    if random.random() < 0.5:
                        newGenome.genes.append(self.genes[geneIndex].copy())

                    else:
                        newGenome.genes.append(other.genes[geneIndex].copy())

                elif self.genes[geneIndex].innovation < other.genes[otherIndex].innovation:
                    if self.organism.fitness >= other.organism.fitness:
                        newGenome.genes.append(self.genes[geneIndex].copy())

                    geneIndex += 1

                else:
                    if self.organism.fitness <= other.organism.fitness:
                        newGenome.genes.append(self.genes[geneIndex].copy())

                    otherIndex += 1

            if self.organism.fitness > other.organism.fitness:
                for gene in self.genes[geneIndex:]:
                    newGenome.genes.append(gene.copy())

            elif self.organism.fitness == other.organism.fitness:
                for gene in self.genes[geneIndex:]:
                    newGenome.genes.append(gene.copy())

                for gene in other.genes[otherIndex:]:
                    newGenome.genes.append(gene.copy())

            else:
                for gene in other.genes[otherIndex:]:
                    newGenome.genes.append(gene.copy())

            return newGenome

        else:
            if self.organism.fitness >= other.organism.fitness:
                return self.copy()

            else:
                return other.copy()

    def sortGenes(self):
        if len(self.genes) == 0:
            return

        print(self.genes)
        orderGenes = [self.genes[0]]
        for gene in self.genes[1:]:
            for i in range(len(orderGenes)):
                if gene.innovation < orderGenes[i].innovation:
                    orderGenes.insert(i, gene)
                    break

                if i == len(orderGenes) - 1:
                    orderGenes.append(gene)

    def randomNeuron(self, useInput):
        neurons = []
        for gene in self.genes:
            if gene.input not in neurons:
                if useInput and gene.input in self.parameters["inputs"] or gene.input not in self.parameters["inputs"]:
                    neurons.append(gene.input)

            if gene.output not in neurons:
                if not useInput and gene.output in self.parameters["output"] or gene.output not in self.parameters["outputs"]:
                    neurons.append(gene.output)

        return random.choice(neurons)


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
        newGene = Gene(self.input, self.output, self.innovation)
        newGene.weight = self.weight
        newGene.age = self.age

        return newGene

class Network:
    def __init__(self, genome):
        self.recurrent = genome.parameters["recurrent"]
        self.steepness = genome.parameters["steepness"]
        self.neurons = {"bias": Neuron(self, 0)}

        for input in genome.parameters["inputs"]:
            self.neurons[input] = Neuron(self, 0)

        genome.sortGenes()
        for gene in genome.genes:
            if gene.enabled:
                if gene.input not in list(self.neurons.keys()):
                    self.neurons[gene.input] = Neuron(self, 0)

                if gene.output not in list(self.neurons.keys()):
                    self.neurons[gene.output] = Neuron(self, self.neurons[gene.input].timeStep + 1)

                if self.neurons[gene.input].timeStep > self.neurons[gene.output].timeStep and self.recurrent:
                    gene.recurrent = True
                    self.neurons[gene.output].sources.append(gene)

                elif self.neurons[gene.input].timeStep == self.neurons[gene.output].timeStep:
                    gene.output.timeStep += 1
                    self.neurons[gene.output].sources.append(gene)

                else:
                    self.neurons[gene.output].sources.append(gene)

    def eval(self, inputs):
        print(self.neurons)
        for key in list(inputs.keys()):
            self.neurons[key].value = inputs[key]

        for key in self.neurons.keys():
            neuron = self.neurons[key]
            neuron.sigmoid(self.steepness)

        output = {}
        for key in self.neurons.keys():
            output[key] = self.neurons[key]

        return output

class Neuron:
    def __init__(self, network, timeStep):
        self.value = 1
        self.lastValue = 1
        self.sources = []
        self.timeStep = timeStep
        self.network = network

    def sigmoid(self, steepness):
        if len(self.sources) == 0:
            return

        self.lastValue = self.value
        values = []
        for gene in self.sources:
            if gene.recurrent:
                values.append(self.network.neurons[gene.input].lastValue * gene.weight)

            else:
                values.append(self.network.neurons[gene.input].value * gene.weight)

        x = sum(values)
        self.value = (2 / ((steepness ** (-x)) + 1)) - 1

population = Population(10)
organism = population.organisms[0]
network = organism.network
organism.genome.link()
network = Network(organism.genome)
output = network.eval({"A": 1, "B": 1})
print(organism.genome.genes)
for gene in organism.genome.genes:
    print(gene.input, gene.output, gene.weight)

for key in network.neurons.keys():
    print(output[key].value, key)
