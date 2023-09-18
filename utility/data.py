import os
import sys

import yaml


class YAMLDataHandler:
    def __init__(self, file_path):
        self.file_path = file_path

    def write(self, data):
        with open(self.file_path, 'w') as file:
            yaml.dump(data, file)

    def read(self):
        with open(self.file_path, 'r') as file:
            loaded_data = yaml.load(file, Loader=yaml.FullLoader)
        return loaded_data

    def update(self, key, value):
        data = self.read()
        data[key] = value
        self.write(data)

    def delete(self, key):
        data = self.read()
        if key in data:
            del data[key]
            self.write(data)

    def delete_file(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)


class OSDataHandler:
    def write(self, file, path):
        with open(path, "w") as f:
            f.write(file)

    def read(self, path):
        with open(path, "r") as f:
            return f.read()

    def append(self, file, path):
        with open(path, "a+") as f:
            f.write(file)

    def remove(self, path):
        if os.path.exists(path):
            os.remove(path)

    def remove_dir(self, path):
        if os.path.exists(path):
            os.rmdir(path)
