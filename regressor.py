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

new_max =  1
new_min = -1


def normalize_to_1_1(min_a, max_a, y_old):
    return (((y_old - min_a) / (max_a - min_a)) * (new_max - new_min)) + new_min


def denormalize_from_1_1(min_a, max_a, y_new):
    return (((y_new - new_min) / (new_max - new_min)) * (max_a - min_a)) + min_a


def normalize_mk(min_a, max_a, y_new):
    return (2 * (y_new - min_a) / (max_a - min_a)) - 1


def denormalize_mk(min_a, max_a, y_old):
    return ((y_old + 1) / 2) * (max_a - min_a) + min_a


normalized = normalize_to_1_1(-10, 10, 9)
denormalized = denormalize_from_1_1(-10, 10, normalized)

normalized_mk = normalize_mk(-10, 10, 9)
denormalized_mk = denormalize_mk(-10, 10, normalized_mk)

def get_component_value(exponents, inputs):
    value = 1
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

    # loading training set
    BASE_TRAIN_SET = []
    for line in train_set_file_lines:
        train_set_entry = [float(number) for number in line.split(" ")]
        BASE_TRAIN_SET.append(train_set_entry)
    train_set_file.close()

    # # loading inputs
    # INPUT_FROM_STDIN = False
    # INPUTS = copy.deepcopy(BASE_TRAIN_SET)
    # for i in range(len(INPUTS)):
    #     INPUTS[i] = INPUTS[i][:-1]

    INPUT_FROM_STDIN = True
    stdin_input = list(sys.stdin.read().splitlines())
    INPUTS = []
    for line in stdin_input:
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

    TRAINING_VALIDATION_PAIRS = []
    NUMBER_OF_CHECKS = 5

    for i in range(NUMBER_OF_CHECKS):
        base_train_set_copy = copy.deepcopy(BASE_TRAIN_SET)
        random.shuffle(base_train_set_copy)

        current_training_set = []
        current_validation_set = []

        for j in range(len(base_train_set_copy)):
            if j < len(base_train_set_copy) * 0.8:
                current_training_set.append(base_train_set_copy[j])
            else:
                current_validation_set.append(base_train_set_copy[j])

        TRAINING_VALIDATION_PAIRS.append((current_training_set, current_validation_set))

    # PARAMS START
    # MAX_NUMBER_OF_ITERATIONS = 10
    MAX_NUMBER_OF_ITERATIONS = 10000
    EPSILON = 0.001
    LEARNING_RATE = 0.01
    MSE_SCORES = {}

    # best k selection start
    combinations = []
    for k in range(1, 9):
        # print("Processing k == " + k.__str__())
        arr1 = [i for i in range(0, k + 1)]

        # generate combinations of variables
        exponents_all = list(itertools.product(arr1, repeat=NUMBER_OF_COLUMNS - 1))
        EXPONENTS = []
        for exponents in exponents_all:
            if sum(exponents) <= k:
                EXPONENTS.append(exponents)
        avg_mse = 0
        for training_part, validation_part in TRAINING_VALIDATION_PAIRS:
            current_iteration = 0
            weights = [random.uniform(0, 1) for i in range(len(EXPONENTS))]
            while True:
                validation_mse = 0
                for train_set_entry in training_part:
                    adjusted_values = []
                    for exponents in EXPONENTS:
                        adjusted_values.append(get_component_value(exponents, train_set_entry[0:NUMBER_OF_COLUMNS - 1]))
                    prediction = get_actual_value(weights, adjusted_values)
                    diff = prediction - train_set_entry[-1]
                    for i in range(len(weights)):
                        derivative = diff * adjusted_values[i]
                        weights[i] -= LEARNING_RATE * derivative
                for validation_entry in validation_part:
                    adjusted_values_validation = []
                    for exponents in EXPONENTS:
                        adjusted_values_validation.append(
                            get_component_value(exponents, validation_entry[0:NUMBER_OF_COLUMNS - 1]))
                    prediction = get_actual_value(weights, adjusted_values_validation)
                    diff = prediction - validation_entry[-1]
                    validation_mse += (diff * diff)
                validation_mse /= len(validation_part)
                if current_iteration >= MAX_NUMBER_OF_ITERATIONS or validation_mse <= EPSILON:
                    break
                current_iteration += 1
            # print("Validation MSE: " + validation_mse.__str__())
            avg_mse += validation_mse
            if not len(MSE_SCORES) == 0 and (avg_mse / NUMBER_OF_CHECKS) > min(MSE_SCORES.values()):
                avg_mse = NUMBER_OF_CHECKS * validation_mse
                break
        avg_mse /= len(TRAINING_VALIDATION_PAIRS)
        # print("AVG validation MSE: " + avg_mse.__str__())
        MSE_SCORES[k] = avg_mse

    # final model training parameters
    RESULT_MAX_ITERATIONS = 50000
    # RESULT_MAX_ITERATIONS = 50000
    RESULT_LEARNING_RATE = 0.01
    RESULT_EPSILON = 0.0001
    BEST_K = min(MSE_SCORES.items(), key=lambda t: t[1])

    # generate combinations of variables
    result_arr = [i for i in range(0, BEST_K[0] + 1)]
    result_exponents_all = list(itertools.product(result_arr, repeat=NUMBER_OF_COLUMNS - 1))
    RESULT_EXPONENTS = []

    for exponents in result_exponents_all:
        if sum(exponents) <= BEST_K[0]:
            RESULT_EXPONENTS.append(exponents)

    RESULT_CURRENT_ITERATION = 0
    RESULT_WEIGHTS = [random.uniform(-1, 1) for i in range(len(RESULT_EXPONENTS))]
    ITERATIONS_WITHOUT_BETTER_MSE = 0
    BEST_MSE = 1000000
    BEST_MSE_WEIGHTS = []
    while True:
        result_mse = 0
        for train_set_entry in BASE_TRAIN_SET:
            adjusted_values = []
            for exponents in RESULT_EXPONENTS:
                adjusted_values.append(get_component_value(exponents, train_set_entry[:-1]))
            prediction = get_actual_value(RESULT_WEIGHTS, adjusted_values)
            diff = prediction - train_set_entry[-1]
            result_mse += (diff * diff)
            for i in range(len(RESULT_WEIGHTS)):
                derivative = diff * adjusted_values[i]
                RESULT_WEIGHTS[i] -= (RESULT_LEARNING_RATE * derivative)
        result_mse /= len(BASE_TRAIN_SET)
        if result_mse < BEST_MSE:
            BEST_MSE_WEIGHTS = RESULT_WEIGHTS
            ITERATIONS_WITHOUT_BETTER_MSE = 0
        else:
            ITERATIONS_WITHOUT_BETTER_MSE += 1
        if RESULT_CURRENT_ITERATION >= RESULT_MAX_ITERATIONS or result_mse < RESULT_EPSILON or ITERATIONS_WITHOUT_BETTER_MSE == 1000:
            break
        RESULT_CURRENT_ITERATION += 1

    # print("Result MSE: " + result_mse.__str__())
    # print("Best K: " + BEST_K[0].__str__())
    # print("RESULT_CURRENT_ITERATION: " + RESULT_CURRENT_ITERATION.__str__())

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
        prediction = get_actual_value(BEST_MSE_WEIGHTS, adjusted_values)
        predictions.append(prediction)

    for i in range(len(predictions)):
        denormalized_value = denormalize_from_1_1(TRAINING_SET_MIN_MAXS[-1][0], TRAINING_SET_MIN_MAXS[-1][1], predictions[i])
        # diff = denormalized_value - EXPECTED_OUTPUTS[i]
        # print("Diff: " + diff.__str__())
        print(denormalized_value)
