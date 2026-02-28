import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Basic constants
gamma = 0.1
beta = 0.2
alpha = 0.9
l0 = 1
l1 = 5

n = 10
T = 100
# Define rows of probabilities:
state0probs = [1-gamma,0,gamma]
state1probs = [0,1-gamma,gamma]
state2probs = [beta/2, beta/2, 1-beta]

# Make Gamma matrix of probabilities:
Gamma = np.matrix([state0probs, state1probs, state2probs])

C1 = 2
Clist = [C1]
Zlist = []

random.seed(0)

# Probability of moving forward
def moveForward(state):
    if state == 0:
        return 0 if random.random() < state0probs[0] else 2
    elif state == 1:
        return 1 if random.random() < state1probs[1] else 2
    else:
        return 0 if random.random() < state2probs[0] else (1 if random.random() < state2probs[1] else 2)

# Simulate the for n neurons, starting (always) from state 2, and save the resulting states after T time-steps
startstate = 2
cstatelist = []
cstateslist = []
for i in range(n):
    state = startstate
    # Move n steps forward and save state
    for t in range(T):
        state = moveForward(state)
        cstatelist.append(state)
        #print(f"Neuron {i}, Time {t}, State: {state}")
    # Add the state to the list
    cstateslist.append(cstatelist)
zlists = []
# Generate z values for each neuron based on the observed states in Clist
for i in range(n):
    Zlist = []
    for c in cstateslist[i]:
        match c:
            case 0:
                if random.random() < 1-alpha:
                    Zlist.append(1)
                else:
                    Zlist.append(0)
            case 1:
                if random.random() < alpha:
                    Zlist.append(1)
                else:
                    Zlist.append(0)
            case 2:
                if random.random() < 0.5:
                    Zlist.append(1)
                else:
                    Zlist.append(0)
        zlists.append(Zlist)

# Add lambda for each neuron based on the z values
lambdalists = []
for i in range(n):
    lambdalist = [l0 if z == 0 else l1 for z in zlists[i]]
    lambdalists.append(lambdalist)

# Generate Poisson samples based on the mean of the lambda values
meanlambdalist = np.mean(lambdalists[0])  # Use the first neuron's lambda list for mean calculation
Xlists = []
for i in range(n):
    Xlist = np.random.poisson(lam=meanlambdalist, size=T)
    Xlists.append(Xlist)
print("Poisson samples: ", Xlist)

# Plot X values for different neurons in different subplots
plt.figure(figsize=(12, 6))
for i in range(n):
    plt.subplot(n, 1, i+1)
    plt.plot(Xlists[i], label=f'X Neuron {i}')
    plt.scatter(range(len(Xlists[i])), Xlists[i], color='blue', label=f'X Neuron {i}', alpha=0.5)
    plt.title(f'Poisson Samples (X) for Neuron {i}')
plt.show()

# Plot mean of all neurons with mean curve and scatter plot
mean_Xlist = np.mean(Xlists, axis=0)
plt.figure(figsize=(12, 6))
plt.plot(mean_Xlist, label='Mean X across Neurons')
plt.scatter(range(len(mean_Xlist)), mean_Xlist, color='blue', label='MeanX across Neurons', alpha=0.5)
plt.title('Mean Poisson Samples (X) across Neurons')
plt.legend()
plt.show()