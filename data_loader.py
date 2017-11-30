import csv
import numpy as np
# load in data from file

def create_dictionaries(equation_strings):
    dictionary = {}
    reverse_dictionary = {}
    index = 0
    for equation in equation_strings:
        for element in equation:
            if element not in dictionary.keys():
                dictionary[element] = index
                reverse_dictionary[index] = element
                index += 1
    return dictionary, reverse_dictionary

def eq_strings_to_one_hot(equation_strings, dictionary):
    one_hot_list = []
    for eq_str in equation_strings:
        M = len(eq_str)
        N = len(dictionary)+1
        one_hot = np.zeros((M,N))
        for eq_index, eq_element in enumerate(eq_str):
            one_hot_index = dictionary[eq_element]
            one_hot[eq_index,one_hot_index] = 1
        one_hot_list.append(one_hot)
    return one_hot_list

def load_data(fname_phi,fname_eq):

    equation_strings = []
    with open(fname_eq, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\n')
        for row in reader:
            equation_strings.append(row[0].split(',')+['<eoe>'])
    phi_list = []
    with open(fname_phi, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\n')
        for row in reader:
            phi = row[0].split(',')
            phi_list.append([float(p) for p in phi])

    dictionary, reverse_dictionary = create_dictionaries(equation_strings)
    one_hot_list = eq_strings_to_one_hot(equation_strings, dictionary)
    num_classes = len(dictionary)+1
    num_training = len(equation_strings)
    max_len_equation = max([len(e) for e in equation_strings])
    return phi_list, equation_strings, one_hot_list, dictionary, reverse_dictionary
