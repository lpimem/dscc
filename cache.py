import pickle
from . import get_cache_filename

def save(k, v):
    with open(get_cache_filename(k), 'wb') as f:
        pickle.dump(v, f)
        print(k, "saved")

def load(k):
    try:
        with open(get_cache_filename(k), 'rb') as f:
            return pickle.load(f)
    except:
        print("Cannot load cache for", k)
        return None

def main():
    print(get_cache_filename("abc"))

if __name__ == '__main__':
    main()