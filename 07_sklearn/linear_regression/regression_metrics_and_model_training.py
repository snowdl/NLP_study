import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path='../../12_data/USA_Housing.csv'):
    # Load dataset from the specified CSV file path
    df = pd.read_csv(path)
    return df

def explore_data(df):
    # Display first few rows of the dataframe
    print(df.head())
    # Display summary info about dataframe columns and data types
    print(df.info())
    # Show basic statistics of numeric columns
    print(df.describe())

    # Plot pairwise relationships (scatter plots) on a sample of 500 rows for performance
    sns.pairplot(df.sample(500))
    plt.show()

    # Plot distribution of the target variable 'Price'
    sns.displot(df['Price'])
    plt.show()

    # Plot heatmap of correlation matrix between numeric features
    sns.heatmap(df.corr(numeric_only=True), annot=True)
    plt.show()

def train_and_evaluate(X, y, test_size=0.4, random_state=101):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize and train linear regression model
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    # Predict target variable on test set
    predictions = lm.predict(X_test)

    # Calculate evaluation metrics
    mae = metrics.mean_absolute_error(y_test, predictions)  # Mean Absolute Error
    mse = metrics.mean_squared_error(y_test, predictions)   # Mean Squared Error
    rmse = np.sqrt(mse)                                      # Root Mean Squared Error

    # Print model intercept and coefficients
    print('Intercept:', lm.intercept_)
    print('Coefficients:', dict(zip(X.columns, lm.coef_)))
    # Print evaluation metrics
    print(f'MAE: {mae:.3f}')
    print(f'MSE: {mse:.3f}')
    print(f'RMSE: {rmse:.3f}')

    # Plot actual vs predicted prices
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.show()

    # Plot residuals distribution
    sns.histplot(y_test - predictions, bins=30, kde=True)
    plt.title('Residuals Distribution')
    plt.show()

    return lm, X_test, y_test, predictions

def main():
    # Load data
    df = load_data()
    # Explore data visually and statistically
    explore_data(df)

    # Prepare feature matrix X and target vector y
    X = df.drop(columns=['Price', 'Address'])
    y = df['Price']

    # Train model and evaluate performance
    train_and_evaluate(X, y)

if __name__ == "__main__":
    main()
