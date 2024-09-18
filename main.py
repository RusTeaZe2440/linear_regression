import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Experience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.7, 3.9, 4.0, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0, 6.8, 7.1, 7.9],
    'Salary': [39500, 40000, 41600, 43525, 45891, 50642, 55150, 56445, 57189, 60218, 61794, 66957, 70081, 72111, 77938,
               79029, 80088, 81363, 93940, 101302]

}
df = pd.DataFrame(data)

X = df['Experience'].values.reshape(-1, 1)
y = df['Salary']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print(f'Mean Squared error: {mean_squared_error(y, y_pred)}')
print(f'r2 Score: {r2_score(y, y_pred)}')


def Visualization(x: pd.Series, y: pd.Series, pred) -> None:
    plt.style.use('ggplot')
    plt.scatter(x, y, color='orange', edgecolor='blue', marker='o', label='Actual Salary')
    plt.plot(x, pred, color='blue', linewidth='2', label='Predicted Salary')
    plt.title('Simple Linear Regression')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()


Visualization(X, y, y_pred)
