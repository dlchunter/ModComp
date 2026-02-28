import random
import numpy as np
import matplotlib.pyplot as plt


gamma = 0.1
beta = 0.2
alpha = 0.9
l0 = 1
l1 = 5

state0probs = [1-gamma,0,gamma]
state1probs = [0,1-gamma,gamma]
state2probs = [beta/2, beta/2, 1-beta]

Gamma = np.matrix([state0probs, state1probs, state2probs])

n = 10
T = 100

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


# Simulate the process T times, starting (always) from state 2, and save the resulting states after n steps
startstate = 2
for i in range(T):
    state = startstate
    # Move n steps forward and save state
    for t in range(n):
        state = moveForward(state)
    # Add the state to the list
    Clist.append(state)

# Generate z values based on the observed states in Clist
for c in Clist:
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

# Generate lambda values based on the z values
lambdalist = [l0 if z == 0 else l1 for z in Zlist]
# Generate Poisson samples based on the mean of the lambda values
meanlambdalist = np.mean(lambdalist)
Xlist = [np.random.poisson(lambd) for lambd in lambdalist]
print("Poisson samples: ", Xlist)

# Visualize all samples of X,Z,C in one plot
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(Xlist, label='X')
plt.title('Poisson Samples (X)')
plt.subplot(3, 1, 2)
plt.plot(Zlist, label='Z', color='orange')
plt.title('Z Values')
plt.subplot(3, 1, 3)
plt.plot(Clist, label='C', color='green')
plt.title('C Values')
plt.tight_layout()
plt.show()

# Plot c-values on top of the x-values
plt.figure(figsize=(12, 6))
plt.plot(Xlist, label='X')
plt.scatter(range(len(Clist)), Clist, color='red', label='C',
            alpha=0.5)
plt.title('Poisson Samples (X) with C Values')
plt.legend()
plt.show()
