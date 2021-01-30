import getopt
import sys
import copy
import random
import itertools


def get_actual_value(input, weights):
    actual_value = weights[0]
    for i in range(1, len(input)):
        actual_value += weights[i] * input[i]
    return actual_value


def normalize_to_0_1(min, max, value):
    return (value - min) / (max - min)


def denormalize_from_0_1(min, max, normalized_value):
    return (normalized_value * (max - min)) + min


def get_component_value(exponents, inputs):
    value = 1
    assert (len(exponents) == len(inputs))
    for i in range(len(exponents)):
        value *= pow(inputs[i], exponents[i])

    return value


def get_output_value(weights, inputs):
    value = 0
    assert (len(weights) == len(inputs))
    for i in range(len(weights)):
        value += weights[i] * inputs[i]
    return value


if __name__ == '__main__':
    options, values = getopt.getopt(sys.argv[1:], "t:")
    t_option_value = None

    for key, value in options:
        if key == '-t':
            t_option_value = value

    # train set loading
    train_set_file = open(t_option_value, 'r')
    train_set_file_lines = train_set_file.readlines()

    BASE_TRAIN_SET = []

    for line in train_set_file_lines:
        train_set_entry = [float(number) for number in line.split(" ")]
        BASE_TRAIN_SET.append(train_set_entry)
    train_set_file.close()
    # train set loading end

    INPUT_FROM_STDIN = False
    INPUTS = copy.deepcopy(BASE_TRAIN_SET)
    for input_entry in INPUTS:
        input_entry = input_entry[:-1]

    INPUT_FROM_STDIN = True
    stdin_input = list(sys.stdin.read().splitlines())
    INPUTS = []
    for line in stdin_input:
        INPUTS.append([float(item) for item in line.split(' ')])

    assert (not len(BASE_TRAIN_SET) == 0)
    NUMBER_OF_COLUMNS = len(BASE_TRAIN_SET[0])
    for train_set_entry in BASE_TRAIN_SET:
        assert (len(train_set_entry) == NUMBER_OF_COLUMNS)

    # randomize training and validation sets, collect min and max values start
    train_and_validation_pairs = []
    min_maxs = []
    for i in range(1):
        mins = []
        maxs = []

        for j in range(NUMBER_OF_COLUMNS):
            maxs.append(-100000000)
            mins.append(100000000)

        train_set_copy = copy.deepcopy(BASE_TRAIN_SET)
        random.shuffle(train_set_copy)
        random_train_set = []
        random_validation_set = []

        for j in range(len(train_set_copy)):
            for k in range(NUMBER_OF_COLUMNS):
                if train_set_copy[j][k] < mins[k]:
                    mins[k] = train_set_copy[j][k]
                if train_set_copy[j][k] > maxs[k]:
                    maxs[k] = train_set_copy[j][k]

            if j < (len(train_set_copy) * 0.8):
                random_train_set.append(train_set_copy[j])
            else:
                random_validation_set.append(train_set_copy[j])

        mins_maxs_iteration = []

        for k in range(NUMBER_OF_COLUMNS):
            min_max_pair = (mins[k], maxs[k])
            mins_maxs_iteration.append(min_max_pair)

        min_maxs.append(mins_maxs_iteration)
        train_and_validation_pairs.append((random_train_set, random_validation_set))
    # randomize training and validation sets, collect min and max values end

    # normalize all training and validation sets start
    pair_index = 0
    for train_set, validation_set in train_and_validation_pairs:
        for j in range(len(train_set)):
            for k in range(NUMBER_OF_COLUMNS):
                train_set[j][k] = normalize_to_0_1(min_maxs[pair_index][k][0], min_maxs[pair_index][k][1],
                                                   train_set[j][k])
        for j in range(len(validation_set)):
            for k in range(NUMBER_OF_COLUMNS):
                validation_set[j][k] = normalize_to_0_1(min_maxs[pair_index][k][0], min_maxs[pair_index][k][1],
                                                        validation_set[j][k])
        pair_index += 1
    # normalize all training and validation sets end

    # PARAMS START
    MAX_NUMBER_OF_ITERATIONS = 20
    EPSILON = 0.001
    LEARNING_RATE = 0.01
    MSE_SCORES = {}
    # PARAMS END

    # best k selection start
    combinations = []
    for k in range(1, 9):
        # print("Processing k == " + k.__str__())
        arr1 = [i for i in range(0, k + 1)]
        exponents_list = list(itertools.product(arr1, repeat=NUMBER_OF_COLUMNS - 1))
        for train_set, validation_set in train_and_validation_pairs:
            current_iteration = 0
            weights = [random.uniform(-1, 1) for i in range(len(exponents_list))]
            mse = 0
            while True:
                for train_set_entry in train_set:
                    adjusted_values = []
                    for exponents in exponents_list:
                        adjusted_values.append(get_component_value(exponents, train_set_entry[0:NUMBER_OF_COLUMNS - 1]))
                    assert (len(weights) == len(adjusted_values))
                    prediction = get_actual_value(weights, adjusted_values)
                    diff = prediction - train_set_entry[-1]

                    for i in range(len(weights)):
                        derivative = diff * adjusted_values[i]
                        weights[i] -= LEARNING_RATE * derivative

                for validation_entry in validation_set:
                    adjusted_values_validation = []
                    for exponents in exponents_list:
                        adjusted_values_validation.append(
                            get_component_value(exponents, validation_entry[0:NUMBER_OF_COLUMNS - 1]))
                    assert (len(weights) == len(adjusted_values_validation))
                    prediction = get_actual_value(weights, adjusted_values_validation)
                    diff = prediction - validation_entry[-1]
                    mse += (diff * diff)

                mse /= len(validation_set)
                if current_iteration >= MAX_NUMBER_OF_ITERATIONS or mse <= EPSILON:
                    break
                current_iteration += 1

            # print(mse)
            MSE_SCORES[k] = mse

    result_mins = []
    result_maxs = []

    for j in range(NUMBER_OF_COLUMNS):
        result_maxs.append(-100000000)
        result_mins.append(100000000)

    for j in range(len(BASE_TRAIN_SET)):
        for k in range(NUMBER_OF_COLUMNS):
            if BASE_TRAIN_SET[j][k] < result_mins[k]:
                result_mins[k] = BASE_TRAIN_SET[j][k]
            if BASE_TRAIN_SET[j][k] > result_maxs[k]:
                result_maxs[k] = BASE_TRAIN_SET[j][k]

    result_mins_maxs = []
    for k in range(NUMBER_OF_COLUMNS):
        min_max_pair = (result_mins[k], result_maxs[k])
        result_mins_maxs.append(min_max_pair)

    # normalize all result training data
    for j in range(len(BASE_TRAIN_SET)):
        for k in range(NUMBER_OF_COLUMNS):
            BASE_TRAIN_SET[j][k] = normalize_to_0_1(result_mins_maxs[k][0], result_mins_maxs[k][1],
                                                    BASE_TRAIN_SET[j][k])
        pair_index += 1
    # normalize all training and validation sets end

    RESULT_MAX_ITERATIONS = 40
    RESULT_LEARNING_RATE = 0.01
    RESULT_EPSILON = 0.001
    BEST_K = min(MSE_SCORES.items(), key=lambda t: t[1])

    arr1 = [i for i in range(0, BEST_K[0] + 1)]

    result_exponents_list = list(itertools.product(arr1, repeat=NUMBER_OF_COLUMNS - 1))
    result_current_iteration = 0
    result_weights = [random.uniform(-1, 1) for i in range(len(result_exponents_list))]

    while True:
        for train_set_entry in BASE_TRAIN_SET:
            adjusted_values = []
            for exponents in result_exponents_list:
                adjusted_values.append(get_component_value(exponents, train_set_entry[0:NUMBER_OF_COLUMNS - 1]))
            assert (len(result_weights) == len(adjusted_values))
            prediction = get_actual_value(result_weights, adjusted_values)
            diff = prediction - train_set_entry[-1]
            for i in range(len(result_weights)):
                derivative = diff * adjusted_values[i]
                result_weights[i] -= RESULT_LEARNING_RATE * derivative
        if result_current_iteration >= RESULT_MAX_ITERATIONS or mse <= RESULT_EPSILON:
            break
        result_current_iteration += 1

    for i in range(len(INPUTS)):
        for k in range(len(INPUTS[i])):
            INPUTS[i][k] = normalize_to_0_1(result_mins_maxs[k][0], result_mins_maxs[k][1], INPUTS[i][k])
        if not INPUT_FROM_STDIN:
            INPUTS[i] = INPUTS[i][:-1]

    predictions = []
    for input in INPUTS:
        adjusted_values = []
        for exponents in result_exponents_list:
            adjusted_values.append(get_component_value(exponents, input))
        assert (len(result_weights) == len(adjusted_values))
        prediction = get_actual_value(result_weights, adjusted_values)
        predictions.append(prediction)

    for prediction in predictions:
        print(denormalize_from_0_1(result_mins_maxs[-1][0], result_mins_maxs[-1][1], prediction))
