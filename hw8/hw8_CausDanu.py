import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load data:
x = np.load("x.npy")
y = np.load("y.npy")

X = np.array((np.ones(x.shape), x, np.square(x))).T

# Analytically solve the system of equations:
a1, a2, a3 = np.linalg.lstsq(X, y)[0]
print(a1, a2, a3)


# Solve using RANSAC:
ransac = linear_model.RANSACRegressor(max_trials=1000, residual_threshold=0.05, min_samples=100)
ransac.fit(x.reshape(-1,1), y.reshape(-1,1))
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

print(ransac.estimator_.coef_)

# Plot inputs and outputs:

plt.scatter(x,y)
plt.ylabel("Y")
plt.xlabel("X")
plt.title("Data and its model fit")

plt.plot(x, a3*x*x + a2*x + a1, 'g', label="My params")
plt.plot(x, 2.0*x*x + -4.2*x + 1.0, 'r', label='True params')

plt.scatter(x[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers')
plt.scatter(x[outlier_mask], y[outlier_mask], color='gold', marker='x',
            label='Outliers')

plt.legend()
plt.show()