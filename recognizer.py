from numpy.random import choice
from numpy import array, dot, random, zeros, ones, all

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

# Theshold for activation of neuron
threshold = 0

# Activation function
unit_step = lambda x: -1 if x < threshold else 1

# Classes
categories = ['A','B', 'C', 'D', 'E', 'J', 'K']

# Initialize a matrix of weights (7 x 63)
w = zeros((7, 63))

# Initialize bias input vector for outputs
b = ones(7)

# Learning rate alpha
alpha = 0.2

# Iterations
iterations = 1000

# Iteration count
iteration_count = 0

# Train over 1000 iterations
for i in xrange(iterations):

  # Error count for classification
  error_count = 0

  # Iteration counter
  iteration_count += 1

  # Loop over training data
  for x, expected in training_data:

    # For each output neuron...
    for j in xrange(0, 7):

      # Compute the total input for the jth output neuron
      result = dot(x, w[j]) + b[j]

      # Compute the output of the jth neuron
      output = unit_step(result)

      # Compute the error of the jth neuron
      error = expected[j] - output

      if error != 0:
        # Increment error count
        error_count += 1

        # Update the weight vector for the jth output neuron
        w[j] += alpha * x * error

        # Update the input bias for the jth output neuron
        b[j] += alpha * error

  if error_count == 0:
    break

# Check if we can successfully classify our data
# for x, _ in training_data:
#   for j in xrange(0, 7):
#     result = dot(x, w[j]) + b[j]
#     print '{}: {} -> {}: {}'.format(j, result, unit_step(result), categories[j])
#   print '---------------------------'

def noise(arr, num):
  indices = choice(range(62), num, replace=False)
  for index in indices:
    # Toggle the value
    if arr[index] == -1:
      arr[index] = +1
    else:
      arr[index] = -1
  return arr

# Testing

# 5 Missing Pixels 
testing_data = [
  (noise(array([+1, +1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, +1, +1, +1]), 5), array([+1, -1, -1, -1, -1, -1, -1])),
  (noise(array([+1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, -1]), 5), array([-1, +1, -1, -1, -1, -1, -1])),
  (noise(array([-1, -1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, +1, -1, -1, +1, +1, +1, +1, -1]), 5), array([-1, -1, +1, -1, -1, -1, -1])),
  (noise(array([+1, +1, +1, +1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, -1]), 5), array([-1, -1, -1, +1, -1, -1, -1])),
  (noise(array([+1, +1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1]), 5), array([-1, -1, -1, -1, +1, -1, -1])),
  (noise(array([-1, -1, -1, +1, +1, +1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, +1, +1, -1, -1]), 5), array([-1, -1, -1, -1, -1, +1, -1])),
  (noise(array([+1, +1, +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, -1, +1, +1]), 5), array([-1, -1, -1, -1, -1, -1, +1])),
]

# Attempt to classify noisy data
for x, expected in testing_data:
  predicted = []
  for j in xrange(0, 7):
    result = dot(x, w[j]) + b[j]
    predicted.append(unit_step(result))
    # print '{}: {} -> {}, Expected: {}'.format(j, result, unit_step(result), _)

  print 'Expected: {}, Predicted: {}, Match: {}'.format(expected, predicted, predicted == expected)
  print '-------------------'



