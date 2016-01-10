from numpy import array, copy, zeros
from random import random


def particle_swarm_optimize(error_func,
                            num_parameters: int,
                            max_iterations: int,
                            parameter_init=random,
                            stopping_error: float=0.0001,
                            num_particles: int=10,
                            max_iterations_without_improvement: int=None,
                            c1: float=2.0,
                            c2: float=2.0) -> (array, float):
    """
    try and minimize the error func

    :param max_iterations_without_improvement: If we have this number of consecutive iterations without
        improvement we stop
    :type error_func: [float] -> float
    :param error_func: A function that takes a single argument,
        the parameters array for the particle and returns a float
    :param num_parameters: The number of parameters the error function array expects
    :param max_iterations: The maximum number of iterations before stopping
    :param parameter_init: The function to use for generating the inital parameter weights.
        If not set random 0..1 will be used
    :type parameter_init: () -> float
    :param stopping_error: The returned value of the error_func at which we stop
    :param num_particles: The number of particles to run each iteration
    :param c1: coefficient to weight the local best
    :param c2: coefficient to weight the global best
    :return: tuple of array of winning parameter values, the error of the winning values
    :rtype: (Numpy.array, float)
    """
    if not max_iterations_without_improvement:
        max_iterations_without_improvement = max_iterations

    # initialize the particles
    particles = []
    for i in range(num_particles):
        parameters = array([parameter_init() for _ in range(num_parameters)])
        p = _Particle(parameters, error_func(parameters))
        particles.append(p)

    # let the first particle be the global best
    best = min(particles, key=lambda x: x.error)
    global_best = _Particle(copy(best.parameters), best.error)

    j = 0
    turns_without_improvement = 0
    while j < max_iterations:
        for p in particles:
            velocity = p.velocity + \
                c1 * random() * (p.best - p.parameters) + \
                c2 * random() * (global_best.parameters - p.parameters)
            p.parameters += velocity

            error = error_func(p.parameters)
            if error < p.error:
                p.error = error
                p.best = copy(p.parameters)

            if error < global_best.error:
                global_best.parameters = copy(p.parameters)
                global_best.error = p.error
                turns_without_improvement = 0
                if error < stopping_error:
                    break
        if turns_without_improvement == max_iterations_without_improvement:
            break
        turns_without_improvement += 1

        j += 1

    return global_best.parameters, global_best.error


class _Particle:
    def __init__(self, parameters: array, error: float):
        self.parameters = parameters
        self.best = parameters
        self.error = error
        self.velocity = zeros(len(parameters))