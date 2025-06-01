from fontTools.misc.cython import returns
from sklearn.model_selection import cross_validate, RepeatedKFold,ShuffleSplit, train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import ydf

"""
    These 2 functions will are used to "validate" the data 

    """
def validation_function(Data, X_range, splits, model_type,d):
    df =pd.read_csv(Data)
    #X = df.iloc[:, X_range - 1]
    #y = df.iloc[:, X_range]

    # initial train/test split (unused by CV, but retained)
    #X_train, X_test, y_train, y_test = train_test_split(X, y)

    rkf = ShuffleSplit(n_splits=splits, test_size=0.3, random_state=d)
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





def validation_multiple(n_shuflles):
    avg_rmses = []
    std_rmses = []
    avg_r2 = []
    std_r2 = []
    for i in range(n_shuflles):
        d=39+i
        print(d)
        done=validation_function(
            "embeddings_PRETRAINED.csv",
            3, 10, "RF",d
        )
        avg_rmses.append(done[1])
        std_rmses.append(np.std(done[0]))
        avg_r2.append(done[3])
        std_r2.append(np.std(done[2]))
    mean_rmses= sum(avg_rmses) / len(avg_rmses)
    mean_std_rmse= sum(std_rmses) / len(std_rmses)
    mean_r2=sum(avg_r2)/ len(avg_r2)
    mean_std_r2=sum(std_r2) / len(std_r2)


    return (mean_rmses,mean_std_rmse,mean_r2,mean_std_r2)

tests_total= validation_multiple(10)
print(tests_total)