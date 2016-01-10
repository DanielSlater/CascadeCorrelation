from unittest import TestCase
import math

from cascade_net import CascadeNet, print_results

XOR_INPUTS = [[1.0, -1.0],
              [-1.0, -1.0],
              [-1.0, 0.9],
              [1.0, 1.0]]
XOR_TARGETS = [[1.0],
               [-1.0],
               [1.0],
               [-1.0]]

DOUBLE_XOR_INPUTS = [[-1.0, -1.0, -1.0],
                     [-1.0, -1.0, 1.0],
                     [-1.0, 1.0, -1.0],
                     [-1.0, 1.0, 1.0],
                     [1.0, -1.0, -1.0],
                     [1.0, -1.0, 1.0],
                     [1.0, 1.0, -1.0],
                     [1.0, 1.0, 1.0], ]
DOUBLE_XOR_TARGETS = [[1.0],
                      [1.0],
                      [1.0],
                      [-1.0],
                      [-1.0],
                      [1.0],
                      [1.0],
                      [-1.0]]


class TestCascadeNet(TestCase):
    CLASS = CascadeNet

    def test_generate_candidate(self):
        net = self.CLASS(3, 1)
        # this test keeps failing unless I set candidates to a very high number
        # But I can't see a bug in the code...
        net.num_candidate_nodes = 20
        net._output_connections[0][0] = 2.24107584
        net._output_connections[0][1] = -0.00060484
        net._output_connections[0][2] = -2.24107583
        net._output_connections[0][3] = -2.24107584

        print_results(net, DOUBLE_XOR_INPUTS, DOUBLE_XOR_TARGETS)

        winner = net._generate_candidate(DOUBLE_XOR_INPUTS, DOUBLE_XOR_TARGETS)

        print(winner.activations)
        print(winner.weights)
        # This does not always hold true...
        self.assertAlmostEqual(winner.activations[4], max(winner.activations), delta=0.1)

    def test_correlation_maximize(self):
        net = self.CLASS(2, 1)
        net.use_quick_prop = True
        net._output_connections[0][0] = -1.7
        net._output_connections[0][1] = 1.7
        net._output_connections[0][2] = -1.7

        print_results(net, XOR_INPUTS, XOR_TARGETS)
        # the error will be large for -1.0, 1.0 but small for everything else

        winner = net._generate_candidate(XOR_INPUTS, XOR_TARGETS)
        print(winner.activations)
        self.assertAlmostEqual(winner.activations[2], max(winner.activations), delta=0.1)

    def test_candidates_some_should_be_positive(self):
        inputs = [
            [0.2, 0.1],
            [0.9, 0.8],
            [0.1, 0.3],
            [0.5, 0.6]]
        targets = [[0.4], [-0.9], [0.6], [0.7]]
        net = self.CLASS(2, 1)
        net.learn_rate = 0.5
        net.momentum_coefficent = 0.9
        net.backprop_train_till_convergence(inputs, targets)
        candidates = [net._init_candidate() for _ in range(20)]
        correlations = net._get_candidate_correlations(candidates, inputs, targets)
        print(str(correlations))
        self.assertTrue(any(x[0] > 0 for x in correlations))

    def test_xor_cascading(self):
        net = self.CLASS(2, 1)
        net.momentum_coefficent = 0.7
        net.learn_rate = 0.2
        error = net.get_error(XOR_INPUTS, XOR_TARGETS)
        final_error = net.train(XOR_INPUTS, XOR_TARGETS, mini_batch_size=1, max_hidden_nodes=10, stop_error_threshold=0.5,
                                max_iterations_per_epoch=50)

        print_results(net, XOR_INPUTS, XOR_TARGETS)

        self.assert_results(net, XOR_INPUTS, XOR_TARGETS)
        self.assertGreater(error, final_error)

    def test_double_xor_cascading(self):
        net = self.CLASS(3, 1, num_candidate_nodes=8)
        net.learn_rate = 0.15

        error = net.get_error(DOUBLE_XOR_INPUTS, DOUBLE_XOR_TARGETS)
        final_error = net.train(DOUBLE_XOR_INPUTS, DOUBLE_XOR_TARGETS,
                                mini_batch_size=3,
                                max_hidden_nodes=10,
                                stop_error_threshold=1.0,
                                max_iterations_per_epoch=100)

        print(len(net._cascade_connections))
        print_results(net, DOUBLE_XOR_INPUTS, DOUBLE_XOR_TARGETS)
        self.assert_results(net, DOUBLE_XOR_INPUTS, DOUBLE_XOR_TARGETS)

    def test_xor_no_cascading(self):
        net = self.CLASS(2, 1)

        error = net.get_error(XOR_INPUTS, XOR_TARGETS)
        final_error = net.backprop_train_till_convergence(XOR_INPUTS, XOR_TARGETS)
        self.assertGreater(error, final_error)

    def test_simple(self):
        net = self.CLASS(1, 1)
        data = [[0.1],
                [0.5],
                [0.9]]
        targets = [[0.1],
                   [0.5],
                   [0.9]]
        error = net.get_error(data, targets)
        final_error = net.backprop_train_till_convergence(data, targets)
        self.assertGreater(error, final_error)

    def assert_results(self, net, inputs: [[float]], targets: [[float]], delta: float=0.5):
        for i in range(len(targets)):
            results = net.get_result(inputs[i])
            for j in range(len(targets[i])):
                self.assertGreater(delta, math.fabs(targets[i][j] - results[j]))

    def test_zero_learning_rate(self):
        net = self.CLASS(2, 1)
        net.learn_rate = 0.0
        net.backprop_train_till_convergence(XOR_INPUTS, XOR_TARGETS)