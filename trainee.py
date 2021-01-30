import getopt
import sys


def calculate_value_of_polynomial(polynomial_factors_input_arg):
    sum_of_components = 0.0
    for component in components:
        value_of_line = component.get(0)
        for index in range(1, n + 1):
            value_of_line *= (pow(polynomial_factors_input_arg[index - 1], component.get(index, 0)))
        sum_of_components += value_of_line
    return sum_of_components


def load_contents_of_description_file(description_file):

    # loading file and splitting contents by spaces
    with open(description_file, 'r') as file_handle:
        read_description = file_handle.read().splitlines()
    read_description = [line.split(" ") for line in read_description]

    # loading n and k values
    value_of_n = int(read_description[0][0])
    value_of_k = int(read_description[0][1])

    component_list = []
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
        component_list.append(variables_occurrences)
    return value_of_n, value_of_k, component_list


if __name__ == '__main__':
    # loading argument value
    description = ""
    options, values = getopt.getopt(sys.argv[1:], "d:")
    for option, value in options:
        if option in ['-d']:
            description = value

    # loading description file contents
    n, k, components = load_contents_of_description_file(description)

    # loading contents of stdin
    polynomial_factors_inputs_from_stdin = list(sys.stdin.read().splitlines())

    # computing output for each input line
    for polynomial_factors_input in polynomial_factors_inputs_from_stdin:
        polynomial_factors_input = [float(number) for number in polynomial_factors_input.split(" ")]
        result = calculate_value_of_polynomial(polynomial_factors_input)
        print(result.__str__())
