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

random.seed(0)

def moveForward(state):
    if state == 0:
        return 0 if random.random() < state0probs[0] else 2
    elif state == 1:
        return 1 if random.random() < state1probs[1] else 2
    else:
        return 0 if random.random() < state2probs[0] else (1 if random.random() < state2probs[1] else 2)

startstate = 2
for i in range(T):
    state = startstate
    for t in range(n):
        state = moveForward(state)
    Clist.append(state)
print(Clist)