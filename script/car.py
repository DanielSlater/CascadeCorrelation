import random
from cascade import CascadeNetwork


def loadData(minValue, maxValue):
    range = maxValue - minValue

    def parseMaint(str):
        if str == "low":
            return minValue
        if str == "med":
            return minValue + range / 3.0
        if str == "high":
            return minValue + range / 1.5
        if str == "vhigh":
            return maxValue
        else: raise Exception("Unknown value")

    def parseDoors(str):
        if str == "5more":
            return maxValue
        else:
            v = float(str) #range 2-5
            v = (v-2.0)/4.0 #now in range 0 to 3/4
            return v*range-minValue

    def parsePersons(str):
        if str == "more":
            return maxValue
        if str == "2":
            return minValue
        if str == "4":
            return minValue + range/2.0

    def parseBoot(str):
        if str == "small": return minValue
        if str == "med": return minValue + range/2.0
        if str == "big": return maxValue

    def parseSafety(str):
        if str == "low": return minValue
        if str == "med": return minValue + range/2.0
        if str == "high": return maxValue

    def parseLabel(str):
        if str == "unacc\n": return minValue
        if str == "acc\n": return minValue + range/3.0
        if str == "good\n": return minValue + range/1.5
        if str == "vgood\n": return maxValue

    inputs = []
    targets = []
    with open('..\\data\\car.data') as f:
        for line in f:
            parts = line.split(',')
            #cols in car.names

            inputs.append([parseMaint(parts[0]),
                           parseMaint(parts[1]),
                           parseDoors(parts[2]),
                           parsePersons(parts[3]),
                           parseBoot(parts[4]),
                           parseSafety(parts[5])])
            targets.append([parseLabel(parts[6])])

    return inputs, targets

def orderbyRand(inputs, targets):
    seq = zip(inputs, targets)
    seq = sorted(seq, key=lambda x: random.random())
    newInputs = []
    newTargets = []
    for x, y in seq:
        newInputs.append(x)
        newTargets.append(y)
    return newInputs, newTargets

inputs, targets = orderbyRand(*loadData(-1.0, 1.0))
trainI = inputs[:1000]
trainT = targets[:1000]
testI = inputs[1001:]
testT = targets[1001:]

net = CascadeNetwork(6, 1, num_candidate_nodes=8, train_candidates_pso=True)
net.learn_rate = 0.05
net.momentum_coefficent = 0.0
net.output_connection_dampening = 1.0

final_error = net.train(trainI, trainT, max_hidden_nodes=20, mini_batch_size=6, stop_error_threshold=0.05)

