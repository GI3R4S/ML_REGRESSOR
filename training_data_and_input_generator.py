import getopt
import sys
import numpy as np
import random

if __name__ == '__main__':

    training_data = []
    inputs_training_range = []
    inputs_wider_range = []
    number_of_points = 200
    
    training_min_arg = -10
    training_max_arg = 10

    training_min_arg_wider = -100
    training_max_arg_wider = 100

    x_1 = 0
    x_2 = 0
    x_3 = 0
    x_4 = 0
    # let's generate data for function 5x_1^4 - 2(x_2^3)(x_3^3) - 4(x_3^2) + 7x_4
    while len(training_data) < number_of_points:
        x_1 = np.random.uniform(training_min_arg, training_max_arg)
        x_2 = np.random.uniform(training_min_arg, training_max_arg)
        x_3 = np.random.uniform(training_min_arg, training_max_arg)
        x_4 = np.random.uniform(training_min_arg, training_max_arg)

        result = 0
        result += (5 * pow(x_1, 4))
        result -= (2 * pow(x_2, 3) * pow(x_3, 3))
        result -= (4 * pow(x_3, 2))
        result += (7 * x_4)

        line = x_1.__str__() + ' ' + x_2.__str__() + ' ' + x_3.__str__() + ' ' + x_4.__str__() + ' ' + result.__str__()
        training_data.append(line)

    # let's generate inputs for function 5x_1^4 - 2(x_2^3)(x_3^3) - 4(x_3^2) + 7x_4 in training range
    while len(inputs_training_range) < number_of_points:
        x_1 = np.random.uniform(training_min_arg, training_max_arg)
        x_2 = np.random.uniform(training_min_arg, training_max_arg)
        x_3 = np.random.uniform(training_min_arg, training_max_arg)
        x_4 = np.random.uniform(training_min_arg, training_max_arg)

        line = x_1.__str__() + ' ' + x_2.__str__() + ' ' + x_3.__str__() + ' ' + x_4.__str__()
        inputs_training_range.append(line)

    # let's generate inputs for function 5x_1^4 - 2(x_2^3)(x_3^3) - 4(x_3^2) + 7x_4 in wider  range
    while len(inputs_wider_range) < number_of_points:
        x_1 = np.random.uniform(training_min_arg_wider, training_max_arg_wider)
        x_2 = np.random.uniform(training_min_arg_wider, training_max_arg_wider)
        x_3 = np.random.uniform(training_min_arg_wider, training_max_arg_wider)
        x_4 = np.random.uniform(training_min_arg_wider, training_max_arg_wider)

        line = x_1.__str__() + ' ' + x_2.__str__() + ' ' + x_3.__str__() + ' ' + x_4.__str__()
        inputs_wider_range.append(line)

    random.shuffle(training_data)
    training_data_output_file = open("training_data_output.txt", 'w')
    for line in training_data:
        training_data_output_file.write(line + '\n')

    training_data_output_file.close()

    inputs_training_range_file = open("inputs_training_range.txt", 'w')
    for line in inputs_training_range:
        inputs_training_range_file.write(line + '\n')

    inputs_training_range_file.close()

    inputs_wider_range_file = open("inputs_wider_range_file.txt", 'w')
    for line in inputs_wider_range:
        inputs_wider_range_file.write(line + '\n')

    inputs_wider_range_file.close()

