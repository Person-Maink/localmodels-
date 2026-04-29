import pickle


def dump(value, file_obj):
    if isinstance(file_obj, str):
        with open(file_obj, "wb") as handle:
            pickle.dump(value, handle)
    else:
        pickle.dump(value, file_obj)


def load(file_obj):
    if isinstance(file_obj, str):
        with open(file_obj, "rb") as handle:
            return pickle.load(handle)
    return pickle.load(file_obj)
