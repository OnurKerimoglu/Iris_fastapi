#import sklearn
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# local modules
from data_funcs import *
from model_funcs import *
from evaluation import *


def main(do_testing=False):
    # load and preprocess data
    X_train, X_test, y_train, y_test = get_data(datasetname='iris')

    # get a model
    model, modelpars = get_model(modelname='randomforestclf')

    # build the pipeline
    pipe = Pipeline(
        [
            ('scalar', StandardScaler()),
            ('model', model)
        ]
    )

    # join  (e.g., preprocessing and model) parameters
    PPpars = {}
    #PPpars = {'scalar__with_mean': [True, False]}
    GSparameters = {**modelpars, **PPpars}

    # GridSearch
    GS = GridSearchCV(pipe, GSparameters)

    # fit (grid search)
    GS.fit(X_train, y_train)
    # print(GS.cv_results_.keys())
    print(
        f'best training score: {GS.best_score_} obtained with {GS.best_params_}')
    best_pipe = GS.best_estimator_

    # save the model in a physical file (e.g., pickle, joblib)
    dump(best_pipe, 'iris_bestpipeline.joblib')  # dump is from the joblib

    # do testing
    if do_testing:
        y_test_hat = best_pipe.predict(X_test)
        # calculate scores
        R2, RMSE, MAE = calc_score(y_test, y_test_hat)
        print(f" R2 = {R2} \n RMSE:{RMSE} \n MAE:{MAE}")

    # return(GS)


if __name__ == '__main__':
    main()
