
def allowed_to_dict(allowed):
    # maybe redundant function, just takes the allowed dictionary and creates a dictionary of indeces (including parentheses) for one hot
    # also makes reverse dictionary, which could be useful in training the RNN
    dictionary = {}
    dictionary['('] = 1
    dictionary[')'] = 2
    dictionary[''] = 3
    for index, a in enumerate(allowed):
        name,eq_class,anon_fn = a
        dictionary[name] = index+4
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary
