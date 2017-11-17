

def allowed_to_dict(allowed):
    dictionary = {}
    dictionary['('] = 1
    dictionary[')'] = 2
    dictionary[''] = 3
    for index, a in enumerate(allowed):
        name,eq_class,anon_fn = a
        dictionary[name] = index+4
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary
