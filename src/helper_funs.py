import json
import pickle


def save_dict_as_json(dictionary, path):
    with open(path, "w") as outfile:
        json.dump(dictionary, outfile)

    return None


def save_obj_as_pickle(obj, path):
    file = open(path, 'wb')
    pickle.dump(obj, file)
    file.close()

    return None


def import_json(path):
    with open(path) as json_file:
        data = json.load(json_file)

    return data


def import_pickle(path):
    # open a file, where you stored the pickled data
    file = open('important', 'rb')

    # dump information to that file
    data = pickle.load(file)

    # close the file
    file.close()

    return data
