from Simple import SimpleTuner

objective_function = lambda vector: -((vector[0] - 0.2) ** 2.0 + (vector[1] - 0.3) ** 2.0) ** 0.5
optimization_domain_vertices = [[0.0, 0.0], [0, 1.0], [1.0, 0.0]]
number_of_iterations = 30
exploration = 0.05# optional, default 0.15

tuner = SimpleTuner(optimization_domain_vertices, objective_function, exploration_preference=exploration)
tuner.optimize(number_of_iterations)
best_objective_value, best_coords = tuner.get_best()

print('Best objective value ', best_objective_value)
print('Found at sample coords ', best_coords)
tuner.plot() # only works in 2D
