__author__ = 'Coleman'

class Environment:
    pass

class Species:
    def __init__(self):
        self.representative = None
        self.organisms = []

class Organism:
    def __init__(self):
        self.genome = None
        self.network = None

class Genome:
    def __init__(self):
        pass

    def copy(self):
        pass

class Gene:
    def __init__(self, innovation):
        pass

    def copy(self):
        pass

class Network:
    def __init__(self, genome):
        self.neurons = {}
        self.outputs = {}
        for input in genome.inputs:
            self.neurons[input] = Neuron()

        for output in genome.outputs:
            self.neurons[output] = Neuron()
            self.outputs[output] = self.neurons[output]

        for gene in genome.genes:
            if gene.enabled:
                if gene.input not in list(self.neurons.keys()):
                    self.neurons[gene.input] = Neuron()

                if gene.output not in list(self.neurons.keys()):
                    self.neurons[gene.input] = Neuron()

                self.neurons[gene.output].inputs.append(gene)

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
    def __init__(self):
        self.value = 0
        self.inputs = []

    def sigmoid(self):
        global steepness
        x = sum([gene.input.value for gene in self.inputs])
        self.value = 2 / (steepness ** x + 1) - 1
        self.inputs = []

currentInnovation = -1
def newInnovation():
    global currentInnovation
    currentInnovation += 1
    return currentInnovation

steepness = 5
distanceThreshold = 0