import getopt
import sys
from random import shuffle


def get_actual_value(input, weights):
    actual_value = weights[0]
    for i in range(1, len(input)):
        actual_value += weights[i] * input[i]
    return actual_value


if __name__ == '__main__':
    options, values = getopt.getopt(sys.argv[1:], "t:i:o:")

    # handle program options
    t_option_value = None
    i_option_value = None
    o_option_value = None

    for key, value in options:
        if key == '-t':
            t_option_value = value
        if key == '-i':
            i_option_value = value
        if key == '-o':
            o_option_value = value
    # handle program options end

    # train set loading
    train_set_file = open(t_option_value, 'r')
    train_set_file_lines = train_set_file.readlines()
    train_set = []
    for line in train_set_file_lines:
        train_set_entry = [float(number) for number in line.split(" ")]
        train_set.append(train_set_entry)
    train_set_file.close()
    # train set loading end

    # iterations loading
    data_in_file = open(i_option_value, 'r')
    data_in_file_line = data_in_file.readline()
    data_in_file_line_splitted = data_in_file_line.split('=')
    maximal_number_of_iterations = int(data_in_file_line_splitted[1])
    data_in_file.close()
    # iterations loading end

    # description loading
    # mock_stdin = \ # testing purposes
    #     [
    #         "1 1",
    #         "1 1.0",
    #         "0 1.0"
    #     ]

    read_description = list(sys.stdin.read().splitlines())
    read_description = [line.split(" ") for line in read_description]

    # loading n and k values
    n = int(read_description[0][0])
    k = int(read_description[0][1])

    description_lines = []
    # for each line
    for description_line in read_description[1:]:
        variables_occurrences = {}
        # for each line's variable
        for variable in description_line[:-1]:
            if int(variable) in variables_occurrences:
                variables_occurrences[int(variable)] += 1
            else:
                variables_occurrences[int(variable)] = 1
        variables_occurrences[0] = float(description_line[-1:][0])
        description_lines.append(variables_occurrences)
    # description loading end

    # parameters
    learning_rate = 0.0115
    epsilon = 0.05
    # parameters end

    # helper variables
    current_iteration = 1
    weights = []
    for description_line in description_lines:
        weights.append(description_line[0])
    # helper variables end

    # gradient calculation
    while current_iteration <= maximal_number_of_iterations:
        derivatives = []
        for i in range(0, n + 1):
            derivatives.append(float(0))

        shuffle(train_set)
        for train_set_entry in train_set:
            inputs = train_set_entry[0:n]
            inputs.insert(0, 1)
            actual_output = get_actual_value(inputs, weights)
            expected_output = train_set_entry[n]
            difference = actual_output - expected_output

            for i in range(len(derivatives)):
                derivatives[i] = difference * inputs[i]
                weights[i] -= (learning_rate * derivatives[i])

        mse = 0
        for train_set_entry in train_set:
            inputs = train_set_entry[0:n]
            inputs.insert(0, 1)
            actual_output = get_actual_value(inputs, weights)
            expected_output = train_set_entry[n]
            difference = actual_output - expected_output
            mse += (difference * difference)
        mse /= len(train_set)

        if mse <= epsilon:
            break
        else:
            current_iteration += 1
    # gradient calculation end 

    # outputting number of iterations
    data_out_file = open(o_option_value, 'w')
    data_out_file.write("iterations=" + (current_iteration -1).__str__())
    data_in_file.close()
    # outputting number of iterations end

    # outputting description
    print(n.__str__() + ' ' + k.__str__())
    for input_index, weight in reversed(list(enumerate(weights))):
        print(input_index.__str__() + ' ' + weight.__str__())
    # outputting description end
