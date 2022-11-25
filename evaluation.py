from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calc_score(y, yhat):
    R2 = r2_score(y, yhat)
    RMSE = mean_squared_error(y, yhat, squared=False)
    MAE = mean_absolute_error(y, yhat)
    return (R2, RMSE, MAE)
