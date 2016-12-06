import os

cache_dir = os.path.sep.join([
    os.path.expanduser("~"), "tmp", "dscc"
])

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

def get_cache_filename(name):
    return os.path.sep.join([cache_dir, name+".p"])