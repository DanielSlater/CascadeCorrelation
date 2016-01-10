from unittest import TestCase
from math import sin, sqrt

from pso import particle_swarm_optimize


class TestPSO(TestCase):
    def test_simplest(self):
        def simple_error(args):
            return args[0]

        result, best = particle_swarm_optimize(simple_error,
                                               1,
                                               100)
        self.assertLess(result[0], 0.1)

    def test_f6(self):
        def f6(parameters):
            para = parameters[0:2]
            numerator = (sin(sqrt((para[0] * para[0]) + (para[1] * para[1])))) * \
                        (sin(sqrt((para[0] * para[0]) + (para[1] * para[1])))) - 0.5
            denominator = (1.0 + 0.001 * ((para[0] * para[0]) + (para[1] * para[1]))) * \
                          (1.0 + 0.001 * ((para[0] * para[0]) + (para[1] * para[1])))
            x = 0.5 - (numerator / denominator)
            return 1 - x

        result, best = particle_swarm_optimize(f6,
                                               2,
                                               100)
        self.assertLess(result[0], 0.1)