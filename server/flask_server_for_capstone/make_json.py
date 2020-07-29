import json
from collections import OrderedDict


def set_data(data, classes):
    file_data = OrderedDict()
    file_data["classes"] = [classes[0], classes[1], classes[2]]
    file_data["predict_number"] = [str(data['p'][0]), str(data['p'][1]), str(data['p'][2])]

    return json.dumps(file_data)