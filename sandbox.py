import pickle
import matplotlib.pyplot as plt
with open('test_histories', 'rb') as f:
    params = pickle.load(f)
    print(params)