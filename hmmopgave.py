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

n = 100
T = 1000
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
for i in range(n):
    state = startstate
    # Move n steps forward and save state
    for t in range(T):
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

X_train, X_test, C_train, C_test = train_test_split(Xlist, Clist, test_size=0.2, random_state=42)
X_train = np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)
# Multiclass logistic regression (automatically induced when logisticregression is used with more than 2 classes) using the X values as features and C values as labels
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, C_train)
predicted_classes = model.predict(X_test)
print("Predicted classes: ", predicted_classes)
print("Classification Report:\n", classification_report(C_test, predicted_classes))
print("Confusion Matrix:\n", confusion_matrix(C_test, predicted_classes))
# Mean squared error asymptotically approaches approx 1.21 as T increases.
mean_squared_error = np.mean((predicted_classes - C_test) ** 2)
print("Mean Squared Error: ", mean_squared_error)

# Plot the predicted classes vs the true classes
plt.figure(figsize=(12, 6))
plt.scatter(range(len(C_test)), C_test, color='blue', label='True Classes',
            alpha=0.5)
plt.scatter(range(len(predicted_classes)), predicted_classes, color='red', label='Predicted Classes', alpha=0.5)
plt.title('True Classes vs Predicted Classes')
plt.legend()
plt.show()
