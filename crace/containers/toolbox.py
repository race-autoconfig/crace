class Toolbox:
    """
    This is a simple implementation of a Toolbox. At the moment it is basically a wrapper for a dictionary.
    It supports setting data through the method set.
    The data can be retrieved by using square brackets.
    """

    def __init__(self):
        self.items = {}

    def __getitem__(self, indices):
        return self.items[indices]

    def __iter__(self):
        return self.items.keys().__iter__()

    def __contains__(self, item):
        return item in self.items.keys()

    def __setitem__(self, key, value):
        self.items[key] = value

    def set(self, name, value):
        self.items[name] = value

