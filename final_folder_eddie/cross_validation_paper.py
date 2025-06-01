from sklearn.model_selection import cross_validate, RepeatedKFold,ShuffleSplit, train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import ydf

def validation_function(Data, X_range, splits, model_type):
    df = pd.read_csv(Data)
    X = df.iloc[:, X_range - 1]
    y = df.iloc[:, X_range]

    # initial train/test split (unused by CV, but retained)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    rkf = ShuffleSplit(n_splits=splits, test_size=0.3, random_state=39)
    rmse_scores = []
    r2_scores  = []

    for train_idx, test_idx in rkf.split(df):
        X_train = df.iloc[train_idx]
        X_test  = df.iloc[test_idx]
        print(f"X_train shape: {X_train.shape[0]} rows, {X_train.shape[1]} columns")
        print(f"X_test shape: {X_test.shape[0]} rows, {X_test.shape[1]} columns")
        if model_type == "RF":
            model = ydf.GradientBoostedTreesLearner(label="Yield",
                                                    task=ydf.Task.REGRESSION) \
                       .train(X_train)

        # evaluate & predict
        _ = model.evaluate(X_test)
        y_pred = model.predict(X_test)

        # compute RMSE
        mse  = mean_squared_error(X_test["Yield"], y_pred)
        rmse = np.sqrt(mse)
        rmse_scores.append(rmse)

        # compute R^2
        r2 = r2_score(X_test["Yield"], y_pred)
        r2_scores.append(r2)

    avg_rmse = sum(rmse_scores) / len(rmse_scores)
    avg_r2   = sum(r2_scores)  / len(r2_scores)

    return rmse_scores, avg_rmse, r2_scores, avg_r2

# example call
tests = validation_function(
    "data_hydrogenation_tokenized_for_token_model.csv",
    3, 10, "RF"
)
print(tests)




