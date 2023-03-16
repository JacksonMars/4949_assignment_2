import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import RFE
import pickle
import matplotlib.pyplot as plt
import seaborn as sb


def setup_dataframe():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    df = pd.read_csv(
        r"C:\Users\jmars\PycharmProjects\4949_assignment_2\cancer patient data sets.csv",
        encoding="ISO-8859-1",
    )
    df.loc[df["Level"] != "High", "Level"] = "Low"

    numeric_not_binned = df[
        [
            "Age",
            "Air Pollution",
            "Alcohol use",
            "Dust Allergy",
            "OccuPational Hazards",
            "Genetic Risk",
            "chronic Lung Disease",
            "Balanced Diet",
            "Obesity",
            "Smoking",
            "Passive Smoker",
            "Chest Pain",
            "Coughing of Blood",
            "Fatigue",
            "Weight Loss",
            "Shortness of Breath",
            "Wheezing",
            "Swallowing Difficulty",
            "Clubbing of Finger Nails",
            "Frequent Cold",
            "Dry Cough",
            "Snoring",
            "Level",
        ]
    ]
    non_numeric = df[["Gender"]]

    dummies = pd.get_dummies(non_numeric, columns=non_numeric.columns)

    df = pd.concat(([numeric_not_binned, dummies]), axis=1)

    return df


def get_test_and_train_data(train_index, test_index, df, X_columns):
    df_train = df.iloc[train_index, :]
    df_test = df.iloc[test_index, :]
    X_train = df_train[X_columns]
    X_test = df_test[X_columns]
    y_train = df_train[["Level"]]
    y_test = df_test[["Level"]]
    return X_train, X_test, y_train, y_test


def test_logistic_model(df, X_columns):
    kfold = KFold(n_splits=5)
    foldCount = 0
    accuracyList = []
    precision_list = []
    recall_list = []
    f1_list = []

    for train_index, test_index in kfold.split(df):
        X_train, X_test, y_train, y_test = get_test_and_train_data(
            train_index, test_index, df, X_columns)

        logistic_model = LogisticRegression(fit_intercept=True, solver="liblinear")
        logistic_model.fit(X_train, y_train)
        y_pred = logistic_model.predict(X_test)
        y_prob = logistic_model.predict_proba(X_test)

        y_test_array = np.array(y_test["Level"])
        cm = pd.crosstab(
            y_test_array, y_pred, rownames=["Actual"], colnames=["Predicted"]
        )

        print("\n***K-fold: " + str(foldCount))
        foldCount += 1

        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracyList.append(accuracy)
        print("\nAccuracy: ", accuracy)
        print("\nConfusion Matrix")
        print(cm)

        FN = cm["Low"]["High"]
        FP = cm["High"]["Low"]
        TP = cm["High"]["High"]

        print("")
        precision = TP / (FP + TP)
        print("\nPrecision:  " + str(round(precision, 3)))
        precision_list.append(precision)

        recall = TP / (TP + FN)
        print("Recall:     " + str(round(recall, 3)))
        recall_list.append(recall)

        F1 = 2 * ((precision * recall) / (precision + recall))
        print("F1:         " + str(round(F1, 3)))
        f1_list.append(F1)

    print("\nAccuracy and Standard Deviation For All Folds:")
    print("*********************************************")
    print("Average accuracy: " + str(np.mean(accuracyList)))
    print("Average precision: " + str(np.mean(precision_list)))
    print("Average recall: " + str(np.mean(recall_list)))
    print("Average F1: " + str(np.mean(f1_list)))



    return {
        "Accuracy": np.mean(accuracyList),
        "Precision": np.mean(precision_list),
        "Recall": np.mean(recall_list),
        "F1": np.mean(f1_list),
    }


def test_rfe_features(num_features, X, y, final_df):
    model = LogisticRegression(fit_intercept=True, solver="liblinear")
    rfe = RFE(estimator=model, n_features_to_select=num_features)
    rfe = rfe.fit(X, y)

    X_columns = []
    for i in range(0, len(X.keys())):
        if rfe.support_[i]:
            X_columns.append(X.keys()[i])

    return test_logistic_model(final_df, X_columns)


def create_plots(df):
    # plt.subplots(nrows=3, ncols=3)
    #
    # plt.subplot(3, 3, 1)
    # wheezing = df['Wheezing']
    # plt.hist(wheezing)
    # plt.title("Wheezing")
    # plt.xlabel("Wheezing")
    # plt.ylabel("Frequency")
    #
    # plt.subplot(3, 3, 2)
    # wheezing = df['Clubbing of Finger Nails']
    # plt.hist(wheezing)
    # plt.title("Clubbing of Finger Nails")
    # plt.xlabel("Clubbing of Finger Nails")
    # plt.ylabel("Frequency")
    #
    # plt.subplot(3, 3, 4)
    # cold = df['Frequent Cold']
    # plt.hist(cold)
    # plt.title("Frequent Cold")
    # plt.xlabel("Frequent Cold")
    # plt.ylabel("Frequency")
    #
    # plt.subplot(3, 3, 5)
    # cough = df['Dry Cough']
    # plt.hist(cough)
    # plt.title("Dry Cough")
    # plt.xlabel("Dry Cough")
    # plt.ylabel("Frequency")
    #
    # plt.subplot(3, 3, 7)
    # snoring = df['Snoring']
    # plt.hist(snoring)
    # plt.title("Snoring")
    # plt.xlabel("Snoring")
    # plt.ylabel("Frequency")
    #
    # plt.subplot(3, 3, 8)
    # age = df['Age']
    # plt.hist(age)
    # plt.title("Age")
    # plt.xlabel("Age")
    # plt.ylabel("Frequency")

    # high_risk = df[df['Air Pollution'] > 1]
    # high_risk = high_risk[high_risk['Alcohol use'] > 1]
    # high_risk = high_risk[high_risk['Dust Allergy'] > 1]
    # high_risk = high_risk[high_risk['OccuPational Hazards'] > 1]
    # high_risk = high_risk[high_risk['Obesity'] > 1]
    # high_risk = high_risk[high_risk['Passive Smoker'] > 1]
    # high_risk = high_risk[high_risk['Chest Pain'] > 1]
    # high_risk = high_risk[high_risk['Coughing of Blood'] > 1]
    # high_risk = high_risk[high_risk['Fatigue'] > 1]
    # high_risk = high_risk[high_risk['Clubbing of Finger Nails'] > 1]
    # high_risk = high_risk[high_risk['Frequent Cold'] > 1]
    # high_risk = high_risk[high_risk['Dry Cough'] > 1]
    # high_risk = high_risk[high_risk['Snoring'] > 1]
    #
    # values = np.array([high_risk['Level'].value_counts()['High'], high_risk['Level'].value_counts()['Low']])
    # labels = ['High', 'Low']
    # plt.pie(values, labels=labels, autopct='%1.0f%%')
    # plt.title("High Risk vs. Low Risk for Patients With Some Factors in Every Category")
    #
    # plt.tight_layout()
    # plt.show()

    corr = df.corr()
    hm = sb.heatmap(corr, cmap="Blues")
    plt.tight_layout()
    plt.show()


df = setup_dataframe()

X = df[
    [
        "Air Pollution",
        "Alcohol use",
        "Dust Allergy",
        "OccuPational Hazards",
        "Genetic Risk",
        "chronic Lung Disease",
        "Balanced Diet",
        "Obesity",
        "Smoking",
        "Passive Smoker",
        "Chest Pain",
        "Coughing of Blood",
        "Fatigue",
        "Weight Loss",
        "Shortness of Breath",
        "Wheezing",
        "Swallowing Difficulty",
        "Clubbing of Finger Nails",
        "Frequent Cold",
        "Dry Cough",
        "Snoring",
        "Gender_1",
        "Age",
    ]
]
y = df[["Level"]]

model = LogisticRegression(fit_intercept=True, solver="liblinear")
rfe = RFE(estimator=model, n_features_to_select=15)
rfe = rfe.fit(X, y)

X_columns = []
for i in range(0, len(X.keys())):
    if rfe.support_[i]:
        X_columns.append(X.keys()[i])
print(X_columns)

test_logistic_model(df, X_columns)

# The below code should be uncommented if you want to generate a pkl file for the model
# results = {}
# for i in range(4, len(X.columns)):
#     result = test_rfe_features(i, X, y, df)
#     results[i] = result
#
# for key in results.keys():
#     print(str(key) + ": " + str(results[key]))

# X_train, X_test, y_train, y_test = train_test_split(X[X_columns], y, test_size=0.25)
# print(X_train.columns)
#
# logistic_model = LogisticRegression(fit_intercept=True, solver="liblinear")
# logistic_model.fit(X_train, y_train)
#
# # Save the model.
# with open('model_pkl', 'wb') as files:
#     pickle.dump(logistic_model, files)
#
# # load saved model
# with open('model_pkl' , 'rb') as f:
#     loadedModel = pickle.load(f)
#
# y_pred = loadedModel.predict(X_test)
# y_prob = loadedModel.predict_proba(X_test)
#
# y_test_array = np.array(y_test["Level"])
# cm = pd.crosstab(
#     y_test_array, y_pred, rownames=["Actual"], colnames=["Predicted"]
# )
#
# accuracy = metrics.accuracy_score(y_test, y_pred)
# print("\nAccuracy: ", accuracy)
# print("\nConfusion Matrix")
# print(cm)
#
# FN = cm["Low"]["High"]
# FP = cm["High"]["Low"]
# TP = cm["High"]["High"]
#
# print("")
# precision = TP / (FP + TP)
# print("\nPrecision:  " + str(round(precision, 3)))
#
# recall = TP / (TP + FN)
# print("Recall:     " + str(round(recall, 3)))
#
# F1 = 2 * ((precision * recall) / (precision + recall))
# print("F1:         " + str(round(F1, 3)))

# create_plots(df[X_columns])