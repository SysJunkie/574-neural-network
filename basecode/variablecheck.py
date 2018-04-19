import pickle

# Checking variables in the params.pickle file

f = open("params.pickle","rb")
selected_features = pickle.load(f)
n_hidden = pickle.load(f)
w1 = pickle.load(f)
w2 = pickle.load(f)
lambdaval = pickle.load(f)
f.close()
print(selected_features)
print(n_hidden)
print(w1.shape)
print(w2.shape)
print(lambdaval)