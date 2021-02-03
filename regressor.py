import getopt
import sys
import copy
import random
import itertools
import math
import time

new_max = 1
new_min = -1


def normalize_to_1_1(min_a, max_a, y_old):
    return (2 * (y_old - min_a) / (max_a -min_a)) - 1


def denormalize_from_1_1(min_a, max_a, y_new):
    return ((y_new + 1) / 2) * (max_a - min_a) + min_a


def get_component_value(exponents, inputs):
    value = 1
    for i in range(len(exponents)):
        value *= pow(inputs[i], exponents[i])

    return value


def get_actual_value(input, weights):
    actual_value = weights[0]
    for i in range(1, len(input)):
        actual_value += weights[i] * input[i]
    return actual_value


def check_if_is_lesser_than_epsilon(list1, list2, epsilon):
    distance = 0
    for i in range(len(list1)):
        distance += abs(list1[i] - list2[i])
        if distance >= epsilon:
            return False
    return True


if __name__ == '__main__':
    options, values = getopt.getopt(sys.argv[1:], "t:")
    t_option_value = None

    for key, value in options:
        if key == '-t':
            t_option_value = value

    # train set loading
    train_set_file = open(t_option_value, 'r')
    train_set_file_lines = train_set_file.readlines()

    # loading training set
    BASE_TRAIN_SET = []
    for line in train_set_file_lines:
        train_set_entry = [float(number) for number in line.split(" ")]
        BASE_TRAIN_SET.append(train_set_entry)
    train_set_file.close()

    std_in_input = list(sys.stdin.read().splitlines())
    INPUTS = []
    for line in std_in_input:
        INPUTS.append([float(item) for item in line.split(' ')])

    # Loading list of minimal and maximum values
    EXPECTED_OUTPUTS = [float(numbers[-1]) for numbers in BASE_TRAIN_SET]
    NUMBER_OF_COLUMNS = len(BASE_TRAIN_SET[0])

    TRAINING_SET_MIN_MAXS = []
    mins = []
    maxs = []

    for k in range(NUMBER_OF_COLUMNS):
        maxs.append(-100000000)
        mins.append(100000000)

    for j in range(len(BASE_TRAIN_SET)):
        for k in range(NUMBER_OF_COLUMNS):
            if BASE_TRAIN_SET[j][k] < mins[k]:
                mins[k] = BASE_TRAIN_SET[j][k]
            if BASE_TRAIN_SET[j][k] > maxs[k]:
                maxs[k] = BASE_TRAIN_SET[j][k]

    for k in range(NUMBER_OF_COLUMNS):
        min_max_pair = (mins[k], maxs[k])
        TRAINING_SET_MIN_MAXS.append(min_max_pair)

    # normalize all training and validation sets start
    for j in range(len(BASE_TRAIN_SET)):
        for k in range(NUMBER_OF_COLUMNS):
            BASE_TRAIN_SET[j][k] = normalize_to_1_1(TRAINING_SET_MIN_MAXS[k][0], TRAINING_SET_MIN_MAXS[k][1],
                                                    BASE_TRAIN_SET[j][k])

    # PARAMS START
    NUMBER_OF_CHECKS = 4
    MAX_NUMBER_OF_ITERATIONS = 10000
    MINIMUM_DERIVATIVE_DELTA = 1e-8
    LEARNING_RATE = 0.02
    DIVISION_PROPORTION = 0.9

    TRAINING_VALIDATION_PAIRS = []
    MSE_SCORES = {}
    base_train_set_copy = copy.deepcopy(BASE_TRAIN_SET)
    random.shuffle(base_train_set_copy)

    # shuffle sets
    for i in range(NUMBER_OF_CHECKS):
        random_index = random.uniform(0, len(base_train_set_copy) * DIVISION_PROPORTION)
        current_training_set = []
        current_validation_set = []

        for j in range(len(base_train_set_copy)):
            if random_index < j < random_index + len(base_train_set_copy) * (1 - DIVISION_PROPORTION):
                current_validation_set.append(base_train_set_copy[j])
            else:
                current_training_set.append(base_train_set_copy[j])

        TRAINING_VALIDATION_PAIRS.append([current_training_set, current_validation_set])

    # best k selection start
    IS_INVALID = False
    for k in range(1, 9):
        arr1 = [i for i in range(0, k + 1)]

        # generate combinations of exponents
        exponents_all = list(itertools.product(arr1, repeat=NUMBER_OF_COLUMNS - 1))
        EXPONENTS = []
        for exponents in exponents_all:
            if sum(exponents) <= k:
                EXPONENTS.append(exponents)

        total_mse = 0
        pair_index = 0

        IS_INVALID = False
        for training_part, validation_part in TRAINING_VALIDATION_PAIRS:

            weights = [random.random() for i in range(len(EXPONENTS))]
            derivatives = [0 for i in range(len(EXPONENTS))]
            derivatives_prev = []

            adjusted_training_inputs_and_outputs = []

            for i in range(len(training_part)):
                adjusted_values = []
                for exponents in EXPONENTS:
                    adjusted_value = get_component_value(exponents, training_part[i][0:NUMBER_OF_COLUMNS - 1])
                    adjusted_values.append(adjusted_value)

                output = training_part[i][-1]
                pair = (adjusted_values, output)
                adjusted_training_inputs_and_outputs.append(pair)

            for current_iteration in range(MAX_NUMBER_OF_ITERATIONS):
                derivatives_prev = derivatives[:]
                for train_set_entry in adjusted_training_inputs_and_outputs:

                    prediction = get_actual_value(train_set_entry[0], weights)
                    diff = prediction - train_set_entry[1]

                    for i in range(len(weights)):
                        derivatives[i] = diff * train_set_entry[0][i]
                        weights[i] -= LEARNING_RATE * derivatives[i]

                if math.isnan(diff) or math.isinf(diff):
                    IS_INVALID = True
                    break

                if check_if_is_lesser_than_epsilon(derivatives_prev, derivatives, MINIMUM_DERIVATIVE_DELTA):
                    break

            validation_mse = 0
            if IS_INVALID:
                break

            for validation_entry in validation_part:

                adjusted_values_validation = []
                for exponents in EXPONENTS:
                    adjusted_values_validation.append(
                        get_component_value(exponents, validation_entry[0:NUMBER_OF_COLUMNS - 1]))

                prediction = get_actual_value(adjusted_values_validation, weights)
                diff = prediction - validation_entry[-1]
                validation_mse += (diff * diff)

            validation_mse /= len(validation_part)
            pair_index += 1
            total_mse += validation_mse

            if not len(MSE_SCORES) == 0 and total_mse > min(MSE_SCORES.values()):
                total_mse += ((total_mse / pair_index) * (NUMBER_OF_CHECKS - pair_index))
                break

        if IS_INVALID:
            for i in range(k, 9):
                MSE_SCORES[i] = 9999999
            break

        MSE_SCORES[k] = total_mse

    BEST_K = min(MSE_SCORES.items(), key=lambda t: t[1])

    # =======================================================================================
    # =======================================================================================
    # =======================================================================================

    # generate combinations of exponents
    final_training_start = time.time()
    result_arr = [i for i in range(0, BEST_K[0] + 1)]
    result_exponents_all = list(itertools.product(result_arr, repeat=NUMBER_OF_COLUMNS - 1))
    RESULT_EXPONENTS = []

    for exponents in result_exponents_all:
        if sum(exponents) <= BEST_K[0]:
            RESULT_EXPONENTS.append(exponents)

    # final model training parameters
    RESULT_LEARNING_RATE = 0.033
    RESULT_MAX_ITERATIONS = 150000
    RESULT_EPSILON = 1e-10
    RESULT_MIN_MSE = 1e-10
    RESULT_WEIGHTS = [random.random() for i in range(len(RESULT_EXPONENTS))]
    BEST_MSE = 1000000
    BEST_MSE_WEIGHTS = []
    RESULT_CURRENT_ITERATION = 0

    adjusted_result_inputs_and_outputs = []

    for i in range(len(BASE_TRAIN_SET)):
        adjusted_values = []
        for exponents in RESULT_EXPONENTS:
            adjusted_value = get_component_value(exponents, BASE_TRAIN_SET[i][0:NUMBER_OF_COLUMNS - 1])
            adjusted_values.append(adjusted_value)

        output = BASE_TRAIN_SET[i][-1]
        pair = (adjusted_values, output)
        adjusted_result_inputs_and_outputs.append(pair)

    for i in range(RESULT_MAX_ITERATIONS):
        RESULT_CURRENT_ITERATION += 1
        result_mse = 0
        result_derivatives = [0.0] * len(RESULT_WEIGHTS)
        result_derivatives_prev = result_derivatives[:]
        for train_set_entry in adjusted_result_inputs_and_outputs:

            prediction = get_actual_value(train_set_entry[0], RESULT_WEIGHTS)
            diff = prediction - train_set_entry[1]
            result_mse += (diff * diff)

            for j in range(len(RESULT_WEIGHTS)):
                result_derivatives[j] = diff * train_set_entry[0][j]
                RESULT_WEIGHTS[j] -= (RESULT_LEARNING_RATE * result_derivatives[j])

        result_mse /= len(adjusted_result_inputs_and_outputs)

        if result_mse < BEST_MSE:
            BEST_MSE = result_mse
            BEST_MSE_WEIGHTS = RESULT_WEIGHTS
            if BEST_MSE <= RESULT_MIN_MSE:
                break

        if check_if_is_lesser_than_epsilon(result_derivatives, result_derivatives_prev, RESULT_EPSILON):
            break

    # Loading list of minimal and maximum values
    INPUTS_MIN_MAXS = []
    inputs_mins = []
    inputs_maxs = []

    for k in range(NUMBER_OF_COLUMNS - 1):
        inputs_maxs.append(-100000000)
        inputs_mins.append(100000000)

    for j in range(len(INPUTS)):
        for k in range(len(INPUTS[j])):
            if INPUTS[j][k] < inputs_mins[k]:
                inputs_mins[k] = INPUTS[j][k]
            if INPUTS[j][k] > inputs_maxs[k]:
                inputs_maxs[k] = INPUTS[j][k]

    for k in range(NUMBER_OF_COLUMNS - 1):
        min_max_pair = (inputs_mins[k], inputs_maxs[k])
        INPUTS_MIN_MAXS.append(min_max_pair)

    for i in range(len(INPUTS)):
        for k in range(len(INPUTS[i])):
            INPUTS[i][k] = normalize_to_1_1(INPUTS_MIN_MAXS[k][0], INPUTS_MIN_MAXS[k][1], INPUTS[i][k])

    predictions = []
    for input in INPUTS:
        adjusted_values = []
        for exponents in RESULT_EXPONENTS:
            adjusted_values.append(get_component_value(exponents, input))
        prediction = get_actual_value(adjusted_values, BEST_MSE_WEIGHTS)
        predictions.append(prediction)

    for i in range(len(predictions)):
        denormalized_value = denormalize_from_1_1(TRAINING_SET_MIN_MAXS[-1][0], TRAINING_SET_MIN_MAXS[-1][1],
                                                  predictions[i])
        print(denormalized_value)
