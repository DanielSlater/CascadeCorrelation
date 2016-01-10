from math import tanh
from random import random
import numpy
import sys
from cascade_net import CascadeNet, d_tanh, _CandidateNode
from pso import particle_swarm_optimize


class CascadePsoNet(CascadeNet):
    def __init__(self,
                 input_nodes: int,
                 output_nodes: int,
                 function=tanh,
                 d_function=d_tanh,
                 weight_initialization_func=lambda: (random() - 0.3) * 0.6,
                 num_candidate_nodes: int=8):
        super(CascadePsoNet, self).__init__(input_nodes, output_nodes, function=function,
                                                d_function=d_function,
                                                weight_initialization_func=weight_initialization_func,
                                                num_candidate_nodes=num_candidate_nodes)
        self.train_candidates_max_epochs = 1000

    def _generate_candidate(self, inputs, targets):
        errors = self._get_errors(inputs, targets)
        #mean_errors = [sum([x[y] for x in errors]) / len(errors) for y in range(len(errors[0]))]

        activations = []
        for input in inputs:
            self._feed_forward_hidden_nodes(input)
            activations.append(numpy.copy(self._activations[:self._non_output_nodes()]))

        regularizer_coef = 0.01/self._non_output_nodes()

        def cost_func(parameters):
            candidate_activations = []
            for activation in activations:
                candidate_activations.append(self._get_output_node_activation_from_activations(parameters, activation))

            return -sum(CascadeNet._real_correlations(candidate_activations, errors)) \
                    + regularizer_coef*numpy.sum(numpy.abs(candidate_activations))

        parameters, error = particle_swarm_optimize(cost_func,
                                                    self._non_output_nodes(),
                                                    self.train_candidates_max_epochs,
                                                    stopping_error=-sys.float_info.max,
                                                    max_iterations_without_improvement=60,
                                                    parameter_init=self.weight_initialization_func)

        print("best candidate had score %s" % (-error / (len(self._output_connections))))
        winner = _CandidateNode(parameters)
        candidate_activations = []
        for activation in activations:
            candidate_activations.append(self._get_output_node_activation_from_activations(parameters, activation))
        winner.activations = candidate_activations
        winner.score = -error
        return winner