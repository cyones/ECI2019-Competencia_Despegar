
class Logger():
    def __init__(self, path):
        self.file = open(path, 'w', buffering=1)

    def write(self, str, end=""):
        self.file.write(str)
        print(str, end=end)

    def close(self):
        self.file.close()

