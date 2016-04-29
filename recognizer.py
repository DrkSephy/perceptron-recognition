from __future__ import division
from numpy.random import choice
from prettytable import PrettyTable
from numpy import array, dot, random, zeros, ones, all, any, array_equal, sum

def noise(arr, num):
  indices = choice(range(62), num, replace=False)
  for index in indices:
    # Toggle the value
    if arr[index] == -1:
      arr[index] = +1
    else:
      arr[index] = -1
  return arr

def report(trials, arr):
  percentages = []
  for index, val in enumerate(arr):
    percentages.append('{}%'.format(arr[index] / trials * 100))
  return percentages

# A, B, C, D, E, J, K, Z, I, Y, S, Q, G
training_data = [
  (array([-1, -1, +1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, +1, +1, +1]), array([+1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])),
  (array([+1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, -1]), array([-1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])),
  (array([-1, -1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, +1, -1, -1, +1, +1, +1, +1, -1]), array([-1, -1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])),
  (array([+1, +1, +1, +1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, -1]), array([-1, -1, -1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1])),
  (array([+1, +1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1]), array([-1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, -1, -1])),
  (array([-1, -1, -1, +1, +1, +1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, +1, +1, -1, -1]), array([-1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, -1])),
  (array([+1, +1, +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, -1, +1, +1]), array([-1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1])),
  (array([+1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1]), array([-1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1])),
  (array([+1, +1, +1, +1, +1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1]), array([-1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1])),
  (array([+1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1]), array([-1, -1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1])),
  (array([-1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, -1]), array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1])),
  (array([-1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, +1, -1, -1, +1, +1, -1, -1, -1, +1, +1, +1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1, +1, +1, +1, -1]), array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, -1])),
  (array([-1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1]), array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1])),
]

# Number of times to train the network and test
trials = 100

# Noise level
noise_level = 5

# Results
classifications = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Theshold for activation of neuron
threshold = 0

# Activation function
unit_step = lambda x: -1 if x < threshold else 1

# Initialize a matrix of weights (7 x 63)
w = zeros((13, 63))

# Initialize bias input vector for outputs
b = ones(13)

# Learning rate alpha
alpha = 0.2

# Iterations
iterations = 1000

# Iteration count
iteration_count = 0

for trial in xrange(trials):
  # Train over 1000 iterations
  for i in xrange(iterations):

    # Error count for classification
    error_count = 0

    # Iteration counter
    iteration_count += 1

    # Loop over training data
    for x, expected in training_data:

      # For each output neuron...
      for j in xrange(0, 13):

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

  # 5 Missing Pixels 
  testing_data = [
    (noise(array([+1, +1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, +1, +1, +1]), noise_level), array([+1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])),
    (noise(array([+1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, -1]), noise_level), array([-1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])),
    (noise(array([-1, -1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, +1, -1, -1, +1, +1, +1, +1, -1]), noise_level), array([-1, -1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])),
    (noise(array([+1, +1, +1, +1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, -1]), noise_level), array([-1, -1, -1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1])),
    (noise(array([+1, +1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1]), noise_level), array([-1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, -1, -1])),
    (noise(array([-1, -1, -1, +1, +1, +1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, +1, +1, -1, -1]), noise_level), array([-1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, -1])),
    (noise(array([+1, +1, +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, -1, +1, +1]), noise_level), array([-1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1])),
    (noise(array([+1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1]), noise_level), array([-1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1])),
    (noise(array([+1, +1, +1, +1, +1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1]), noise_level), array([-1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1])),
    (noise(array([+1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1]), noise_level), array([-1, -1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1])),
    (noise(array([-1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, -1]), noise_level), array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1])),
    (noise(array([-1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, +1, -1, -1, +1, +1, -1, -1, -1, +1, +1, +1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1, +1, +1, +1, -1]), noise_level), array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, -1])),
    (noise(array([-1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1]), noise_level), array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1])),
  ]

  # Training index
  index = 0

  # Attempt to classify noisy data
  for x, expected in testing_data:
    predicted = []
    for j in xrange(0, 13):
      result = dot(x, w[j]) + b[j]
      predicted.append(unit_step(result))

    if not array_equal(array(expected), array(predicted)):
      classifications[index] += 1

    # Increment training data index
    index += 1

    # print 'Expected: {}, Predicted: {}, Match: {}'.format(expected, predicted, predicted == expected)
    # print '-------------------'

statistics = report(trials, classifications)

table = PrettyTable()
x = PrettyTable(['Total Trials', 'Theshold', 'Noise', 'Class', 'Misclassification Count', 'Misclassification Percentage'])
x.padding_width = 1
x.add_row([trials, threshold, noise_level, 'A', classifications[0], statistics[0]])
x.add_row([trials, threshold, noise_level, 'B', classifications[1], statistics[1]])
x.add_row([trials, threshold, noise_level, 'C', classifications[2], statistics[2]])
x.add_row([trials, threshold, noise_level, 'D', classifications[3], statistics[3]])
x.add_row([trials, threshold, noise_level, 'E', classifications[4], statistics[4]])
x.add_row([trials, threshold, noise_level, 'J', classifications[5], statistics[5]])
x.add_row([trials, threshold, noise_level, 'K', classifications[6], statistics[6]])
x.add_row([trials, threshold, noise_level, 'Z', classifications[7], statistics[7]])
x.add_row([trials, threshold, noise_level, 'I', classifications[8], statistics[8]])
x.add_row([trials, threshold, noise_level, 'Y', classifications[9], statistics[9]])
x.add_row([trials, threshold, noise_level, 'S', classifications[10], statistics[10]])
x.add_row([trials, threshold, noise_level, 'Q', classifications[11], statistics[11]])
x.add_row([trials, threshold, noise_level, 'G', classifications[12], statistics[12]])
print x