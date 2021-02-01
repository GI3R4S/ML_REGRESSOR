import functools
import getopt
import itertools
import math
import operator
import random
import sys


def get_set_file_name_from_argv(argv):
    set_file = ""
    opts, args = getopt.getopt(argv, "t:")
    for opt, arg in opts:
        if opt in ['-t']:
            set_file = arg
    return set_file


def read_set_values_from_file(set_file):
    train_set_inputs = []
    train_set_outputs = []
    with open(set_file, 'r') as reader:
        lines_from_file = reader.read()
        lines = [line for line in lines_from_file.splitlines() if line.strip()]
        for line in lines:
            line = list(map(float, line.split(" ")))
            train_set_inputs.append(line[:-1])
            train_set_outputs.append(line[-1:])
    return train_set_inputs, train_set_outputs


def read_input_values_from_file(in_file):
    input_values = []
    with open(in_file, 'r') as reader:
        lines_from_file = reader.read()
        input_lines = [line for line in lines_from_file.splitlines() if line.strip()]
        for input_line in input_lines:
            input_line = [float(number) for number in input_line.split(" ")]
            input_values.append(input_line)
    return input_values


def read_input_values_from_stdin(in_file):
    input_lines = list(in_file.read().splitlines())
    input_values = []
    for input_line in input_lines:
        input_line = [float(number) for number in input_line.split(" ")]
        input_values.append(input_line)
    return input_values


def calculate_output_value(input_value, coefficients):
    value = coefficients[0]
    for i in range(1, len(coefficients)):
        value += coefficients[i] * input_value[i]
    return value


def check_if_epsilon_reached(previous_gradients, actual_gradients, epsilon_value):
    difference = 0.0
    for i in range(len(previous_gradients)):
        difference += abs(actual_gradients[i] - previous_gradients[i])
        if difference > epsilon_value:
            return False
    return True


def gradient_descent(train_data_input, train_data_output, coefficients, iterations_number, epsilon_value):
    gradients = [0.0] * (len(train_data_input[0]))
    for iteration in range(1, iterations_number + 1):
        prev_gradients = gradients[:]
        for sample in range(len(train_data_output)):
            difference = calculate_output_value(train_data_input[sample], coefficients) - train_data_output[sample]
            for i in range(len(train_data_input[sample])):
                gradients[i] = difference * train_data_input[sample][i]
                coefficients[i] = coefficients[i] - (alpha * gradients[i])
        if check_if_epsilon_reached(prev_gradients, gradients, epsilon_value):
            break
    return coefficients


def normalize_values(values):
    variable_list = [[] for _ in range(len(values[0]))]
    min_value_list = []
    max_value_list = []
    for i in range(len(values)):
        for j in range(len(values[i])):
            variable_list[j].append(values[i][j])
    for i in range(len(variable_list)):
        min_value_list.append(min(variable_list[i]))
        max_value_list.append(max(variable_list[i]))
    for i in range(len(values)):
        for j in range(len(values[i])):
            values[i][j] = (2 * (values[i][j] - min_value_list[j]) / (max_value_list[j] - min_value_list[j])) - 1
    return values, min_value_list, max_value_list


def calculate_mse(actual_output, expected_output):
    mse = 0.0
    for y in range(len(expected_output)):
        mse += math.pow((expected_output[y] - actual_output[y]), 2)
    mse /= len(expected_output)
    return mse


def create_powers_of_coefficients(k):
    powers = []
    powers_product = list(itertools.product(range(k + 1), repeat=n))
    for i in range(len(powers_product)):
        sums = 0
        for j in range(len(powers_product[i])):
            sums += powers_product[i][j]
        if sums <= k:
            powers.append(powers_product[i])
    return powers


def prepare_train_data(input_set, powers_of_coefficients_list):
    prepared_train_data = [[] for _ in range(len(input_set))]
    for m in range(len(input_set)):
        for i in range(len(powers_of_coefficients_list)):
            value = 1.0
            for j in range(n):
                value *= math.pow(input_set[m][j], powers_of_coefficients_list[i][j])
            prepared_train_data[m].append(value)
    return prepared_train_data


def get_best_polynomial_degree(train_data_input, train_data_output, n_sample, iterations_number, epsilon_value):
    ids = range(len(train_data_output))
    random.shuffle(ids)
    number_of_samples_to_validate = len(train_data_output) / n_sample
    k_value = 8
    min_mse_value = None
    best_k_value = 1
    for k in range(1, k_value + 1):
        mse_sum = 0.0
        powers_of_coefficients = create_powers_of_coefficients(k)
        coefficients_list = [0.0] * len(powers_of_coefficients)
        for i in range(n_sample):
            ids_to_train = ids[0:number_of_samples_to_validate * i] + ids[number_of_samples_to_validate * (i + 1):]
            ids_to_validate = ids[number_of_samples_to_validate * i:number_of_samples_to_validate * (i + 1)]
            train_in = map(train_data_input.__getitem__, ids_to_train)
            train_out = map(train_data_output.__getitem__, ids_to_train)
            validation_in = map(train_data_input.__getitem__, ids_to_validate)
            validation_out = map(train_data_output.__getitem__, ids_to_validate)
            for c in range(len(coefficients_list)):
                coefficients_list[c] = random.random()
            train_data_prepared = prepare_train_data(train_in, powers_of_coefficients)
            validation_data_prepared = prepare_train_data(validation_in, powers_of_coefficients)
            coefficients_list = gradient_descent(train_data_prepared, train_out, coefficients_list, iterations_number,
                                                 epsilon_value)
            calculated_outputs = []
            for validation_data in validation_data_prepared:
                calculated_outputs.append(calculate_output_value(validation_data, coefficients_list))
            mse_sum += calculate_mse(calculated_outputs, validation_out)
            if math.isnan(mse_sum):
                break
        if not min_mse_value or mse_sum < min_mse_value:
            min_mse_value = mse_sum
            best_k_value = k
    return best_k_value


if __name__ == '__main__':
    alpha = 0.03
    number_of_iterations = 10000
    epsilon = 0.0000001
    validation_parts = 4

    train_set = get_set_file_name_from_argv(sys.argv[1:])
    train_set_input, train_set_output = read_set_values_from_file(train_set)
    train_set_input, min_values1, max_values1 = normalize_values(train_set_input)
    train_set_output, min_values2, max_values2 = normalize_values(train_set_output)
    train_set_output = functools.reduce(operator.iconcat, train_set_output, [])
    n = len(train_set_input.__getitem__(0))
    best_k = get_best_polynomial_degree(train_set_input, train_set_output, validation_parts, number_of_iterations,
                                        epsilon)
    if best_k <= 4:
        epsilon = 0.00000000001
        number_of_iterations = 25000
    else:
        epsilon = 0.00000001
        number_of_iterations = 20000
    best_powers_of_coefficients = create_powers_of_coefficients(best_k)
    best_coefficients = [0.0] * len(best_powers_of_coefficients)
    for x in range(len(best_coefficients)):
        best_coefficients[x] = random.random()
    train_data_after_preparation = prepare_train_data(train_set_input, best_powers_of_coefficients)
    best_coefficients = gradient_descent(train_data_after_preparation, train_set_output, best_coefficients,
                                         number_of_iterations, epsilon)
    # in_values = read_input_values_from_stdin(sys.stdin)
    in_values = read_input_values_from_file('inputs_training_range3.txt')
    in_values, min_values3, max_values3 = normalize_values(in_values)
    in_values_after_preparation = prepare_train_data(in_values, best_powers_of_coefficients)
    calculated_output_values = []
    for in_value in in_values_after_preparation:
        calculated_output_values.append(calculate_output_value(in_value, best_coefficients))
    for x in range(len(calculated_output_values)):
        calculated_output_values[x] = ((calculated_output_values[x] + 1) / 2) * (max_values2[0] - min_values2[0]) + \
                                      min_values2[0]
        print (calculated_output_values[x])
