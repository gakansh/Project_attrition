import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


def train_and_evaluate_Logregmodel():
    df_loaded = pd.read_csv('hr_employee.csv')
    df2 = pd.read_csv('hr_employee.csv')
    null_values = df_loaded.isnull().sum()
    duplicates = df_loaded.duplicated().sum()




    numerical_cols = df_loaded.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_loaded.select_dtypes(include=['object']).columns
    #for null values
    # Replace null values in numerical columns with mean
    for col in numerical_cols:
        mean_value = df_loaded[col].mean()
        df_loaded[col].fillna(mean_value, inplace=True)

    # Replace null values in categorical columns with mode
    for col in categorical_cols:
        mode_value = df_loaded[col].mode()[0]  # Get the first mode
        df_loaded[col].fillna(mode_value, inplace=True)


    null_values = df_loaded.isnull().sum()

    duplicates = df_loaded.duplicated().sum()

    le = preprocessing.LabelEncoder()

    df = df_loaded

    l = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']

    df[l] = df[l].apply(le.fit_transform)

    ### Outliers using Standard Deviation
    for col in numerical_cols:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]


    scaler = preprocessing.MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled)

    l = (df.columns).tolist()
    df_scaled.columns = l

    df_scaled['Attrition'] = df_scaled['Attrition'].astype('Int64')

    X = df_scaled.drop('Attrition', axis=1)
    y = df_scaled['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logRegmodel = LogisticRegression()
    logRegmodel.fit(X_train, y_train)

    y_pred_log = logRegmodel.predict(X_test)
    accLog = accuracy_score(y_test, y_pred_log)

    mse_rf = mean_squared_error(y_test, y_pred_log)
    r2_rf = r2_score(y_test, y_pred_log)
    print("Accuracy for Logistic Regression Model: ", accLog)

    return mse_rf, r2_rf, accLog


if __name__ == "__main__":
    mse_rf, r2_rf,accuracy_rf = train_and_evaluate_Logregmodel()
    print(f" logistic regression - R^2 Score: {r2_rf}")
    print(f" logistic regression - Mean Squared Error: {mse_rf}")
    print(f" logistic clasification - Accuracy_rf: {accuracy_rf * 100:.2f}%")
