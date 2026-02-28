import random
import numpy as np

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

lambdalist = [l0 if z == 0 else l1 for z in Zlist]
meanlambdalist = np.mean(lambdalist)
Xlist = [np.random.poisson(lambd) for lambd in lambdalist]
print("Poisson samples: ", Xlist)