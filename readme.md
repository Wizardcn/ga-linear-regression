## Train Linear Regression using Genetic Algorithm

This project was a fun and engaging way for me to take a break from studying during my final examination period. As someone who is interested in machine learning, I wanted to explore the application of Genetic Algorithms for optimizing Linear Regression models.

By working on this project, I was able to apply my knowledge of machine learning and optimization techniques in a practical setting. It was exciting to see how the Genetic Algorithm was able to converge to an optimal solution and find the best-fit line for the given dataset.

I hope that this project can inspire others to explore the intersection of machine learning and optimization, and provide a fun and educational experience for anyone interested in this topic.

## Files

This repository contains the following files:

- `linear_regression.py`: This file contains the implementation of the Linear Regression model.
- `natural_selector.py`: This file contains the implementation of the Natural Selector for the Genetic Algorithm.
- `main.py`: This file contains the pipeline function for running the code.

To train the Linear Regression model, run the following command on your terminal in the root of this project:

```shell
python main.py
```

This will run the pipeline function that is to generate the dataset using the following code:

```python
import numpy as np

X = np.arange(100)
noise = np.random.normal(0, 9, size=(X.shape[0],))
y = 1.323 * X + 123.535 + noise
```
And and run the `natural_selector` function using the following code:

```python
solution = natural_selector(
    X.reshape(X.shape[0], 1),
    y,
    population_size=5000,
    num_generations=1000,
    length_of_convergent_list=10,
    mutation_sigma=500,
    fitness_goal=50
)
```

The pipeline function trains the Linear Regression model using Genetic Algorithm and generates a plot of the best-fit line obtained by the Genetic Algorithm.

## Result
Here is an example of the best-fit line obtained by the Genetic Algorithm for the given dataset:

![Best-fit line](./best-fit.jpg)

The blue dots represent the original data points, and the red line represents the best-fit line obtained by the Genetic Algorithm. As we can see, the Genetic Algorithm was able to find a line that closely fits the data points, even in the presence of Gaussian noise.

## Credits
This project was inspired by the following resources:

- [Genetic Algorithms](https://www.geeksforgeeks.org/genetic-algorithms/) by Geeks for Geeks
