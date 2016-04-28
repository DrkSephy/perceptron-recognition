from random import choice
from numpy import array, dot, random, all, zeros

# Unit step function
unit_step = lambda x: -1 if x < 0 else 1

# A, B, C, D, E, J, K
training_data = [
  (array([-1, -1, +1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, +1, +1, +1]), array([+1, -1, -1, -1, -1, -1, -1])),
  (array([+1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, -1]), array([-1, +1, -1, -1, -1, -1, -1])),
  (array([-1, -1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, +1, -1, -1, +1, +1, +1, +1, -1]), array([-1, -1, +1, -1, -1, -1, -1])),
  (array([+1, +1, +1, +1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, -1]), array([-1, -1, -1, +1, -1, -1, -1])),
  (array([+1, +1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1]), array([-1, -1, -1, -1, +1, -1, -1])),
  (array([-1, -1, -1, +1, +1, +1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, +1, +1, -1, -1]), array([-1, -1, -1, -1, -1, +1, -1])),
  (array([+1, +1, +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, -1, +1, +1]), array([-1, -1, -1, -1, -1, -1, +1])),
]

# Initialize a matrix of weights (7 x 63)
w = zeros((7, 63))

print w

# Learning rate alpha
alpha = 0.2

# Iterations
iterations = 1000

# Iteration count
count = 0

# Train over 1000 iterations
for i in xrange(iterations):

  # Error count for classification
  error_count = 0

  # Increment the iteration count
  count += 1

  # Loop over training data
  for x, expected in training_data:

    # For each output neuron...
    for j in xrange(0, 7):
      
      # Compute input to output neuron
      result = dot(x, w[j])

      # Compute the output of the jth neuron
      output = unit_step(result)

      # Compute the error of the jth neuron
      error = expected[j] - output

      if error != 0:
        # Increment error count by 1
        error_count += 1

        # Increment the corresponding weight vector for the jth neuron
        w[j] += alpha * error * x

  if error_count == 0:
    print '{} {} {} : {}'.format('Converged in', str(count) + ' iterations', 'with weights', w)
    break

# # Check if we can successfully classify our data
# for x, _ in training_data:
#   for j in xrange(0, 7):
#     result = dot(x, w[j])
#     print '{}: {} -> {}'.format(j, result, unit_step(result))
#   print '---------------------------'




