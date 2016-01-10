from cascade_net import CascadeNet, print_results
from cascade_pso_net import CascadePsoNet
from test.test_cascade_net import TestCascadeNet, DOUBLE_XOR_INPUTS, DOUBLE_XOR_TARGETS


class TestCascadePsoNet(TestCascadeNet):
    CLASS = CascadePsoNet

    def _get_cascade(self, net):
        net._output_connections[0][0] = 2.24107584
        net._output_connections[0][1] = -0.00060484
        net._output_connections[0][2] = -2.24107583
        net._output_connections[0][3] = -2.24107584
        print_results(net, DOUBLE_XOR_INPUTS, DOUBLE_XOR_TARGETS)
        return net._generate_candidate(DOUBLE_XOR_INPUTS, DOUBLE_XOR_TARGETS)

    def test_generate_candidate(self):
        super(TestCascadePsoNet, self).test_generate_candidate()

    def test_correlation_pso_vs_derivative(self):
        netPso = self.CLASS(3, 1)
        net = CascadeNet(3, 1)

        candidate = self._get_cascade(net)

        print("Derivative " + str(net._get_candidate_correlations([candidate], DOUBLE_XOR_INPUTS, DOUBLE_XOR_TARGETS)))
        candidate = self._get_cascade(netPso)

        print("PSO " + str(netPso._get_candidate_correlations([candidate], DOUBLE_XOR_INPUTS, DOUBLE_XOR_TARGETS)))