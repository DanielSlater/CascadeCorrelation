from cascade import CascadeNetwork, print_results


def loadData():
    inputs = []
    targets = []
    with open('..\\data\\two_spirals.txt') as f:
        for line in f:
            parts = line.split(' ')
            inputs.append([float(parts[1]),float(parts[3])])
            targets.append([float(parts[4])])
    return inputs, targets

inputs, targets = loadData()

for i in range(len(inputs)):
    for j in range(len(inputs[i])):
        inputs[i][j]=inputs[i][j]*2.0-1.0
for i in range(len(targets)):
    for j in range(len(targets[i])):
        targets[i][j]=targets[i][j]*2.0-1.0

net = CascadeNetwork(2, 1
                     #, train_candidates_pso=True
                     )
net.learn_rate = 0.05
net.momentum_coefficent = 0.0
net.output_connection_dampening = 1.0
net.use_quick_prop = True
#function=logistic, d_function=d_logistic, num_candidate_nodes=40)
CONSTRAINT = 40
#for i in range(10):
final_error = net.train(inputs[:CONSTRAINT], targets[:CONSTRAINT], max_hidden_nodes=25, mini_batch_size=6)
#print_results(net, inputs[:CONSTRAINT], targets[:CONSTRAINT])

correct = 0
total = 0
for i in range(CONSTRAINT):
    result = net.get_result(inputs[i])
    if (result[0] > 0.0) == (targets[i][0] > 0.0):
        correct+=1
    total+=1
print("%s / %s" %(str(correct), str(total)))