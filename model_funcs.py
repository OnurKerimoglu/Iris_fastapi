from sklearn.ensemble import RandomForestClassifier
# GridSearchCV

from sklearn import svm


def get_model(modelname):
    if modelname == 'randomforestclf':
        model = RandomForestClassifier()
        GSparameters = {
            'model__max_depth': [2, 4],
            'model__n_estimators': [75, 100, 125],
            # 'model__min_samples_split': [10, 15, 25],
            'model__criterion': ['gini', 'entropy']
        }
    else:
        raise(ValueError(f'Unkonwn model: {modelname}'))

    return model, GSparameters
