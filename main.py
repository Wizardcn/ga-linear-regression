import numpy as np
from natural_selector import *
import matplotlib.pyplot as plt


def pipeline():

    # Generate random dataset
    X = np.arange(100)
    noise = np.random.normal(0, 9, size=(X.shape[0],))
    y = 1.323 * X + 123.535 + noise

    # train linear regression model using genetic algorithm
    solution = natural_selector(
        X.reshape(X.shape[0], 1),
        y,
        population_size=5000,
        num_generations=1000,
        length_of_convergent_list=10,
        mutation_sigma=500,
        fitness_goal=50
    )

    # visualize result from the regression model
    reg = solution
    y_pred = reg.predict(X.reshape(X.shape[0], 1))

    plt.scatter(X, y)
    plt.plot(y_pred, c="r")

    plt.title(f"Solution: [{reg}]")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend(["actual", "predicted"])

    plt.show()


if __name__ == "__main__":
    pipeline()
