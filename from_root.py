import os

def from_root():
    """
    Returns the absolute path to the project root directory.
    """
    return os.path.dirname(os.path.abspath(__file__))
