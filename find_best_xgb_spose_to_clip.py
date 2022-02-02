import time
import numpy as np
from scipy.io import loadmat, savemat
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV

if __name__=="__main__": 

    regr_data_fn = "data_spose_to_clip.mat"

    start_time = time.time()


    ### load data ### 

    regr_data = loadmat(regr_data_fn)
    x_spose_vecs = regr_data["x_spose"]
    y_clip_vecs = regr_data["y_clip"]


    print("SPoSE shape:", x_spose_vecs.shape)
    print("CLIP shape:", y_clip_vecs.shape)

    xspose_train, xspose_test, yclip_train, yclip_test = train_test_split( x_spose_vecs, 
                                                                           y_clip_vecs, 
                                                                           test_size=0.10, 
                                                                           random_state=42)

    ### run cv ###
    # TODO Sebastian: Use Bayesian CV instead: https://www.kaggle.com/tilii7/bayesian-optimization-of-xgboost-parameters
    # TODO Florians xgb project: https://github.com/florianmahner/Machine-Learning-In-Practice/tree/master/petfinder
    # TODO: multi-output regression exists now: https://xgboost.readthedocs.io/en/latest/tutorials/multioutput.html
    # TODO: regression: https://xgboost.readthedocs.io/en/latest/python/examples/multioutput_regression.html#sphx-glr-python-examples-multioutput-regression-py

    """
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_valid, label=y_valid)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    xgb_params = {'objective' : 'multi:softmax',
                  'eval_metric' : 'mlogloss',
                  'eta' : 0.05,
                  'max_depth' : 4,
                  'num_class' : 5,
                  'lambda' : 0.8
    }

    print('Fitting XGBoost: ')
    bst = xgb.train(xgb_params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=0)
    """


    regressor = xgb.XGBRegressor()
    multiregressor = MultiOutputRegressor( estimator=regressor ) 

    hyparam_grid = {  'estimator__n_estimators': [100, 500, 900, 1100, 1500],
                      'estimator__max_depth': [2, 3, 5, 10, 15],
                      'estimator__learning_rate': [0.05, 0.1, 0.15, 0.20],
                      'estimator__min_child_weight': [1, 2, 3, 4]   }

    random_cv = RandomizedSearchCV( estimator=multiregressor,
                                    param_distributions=hyparam_grid,
                                    cv=5, n_iter=50, n_jobs=32, 
                                    scoring = 'neg_mean_absolute_error', 
                                    verbose = 5, 
                                    return_train_score = True,
                                    random_state=42 )

    random_cv.fit( xspose_train, yclip_train )

    print( "Best estimater:\n", random_cv.best_estimator_ )
    print( "Time elapsed for grid search xgboost SPoSE-to-CLIP: {} minutes.".format( (time.time() - start_time) // 60 ) )    

    import IPython; IPython.embed()

    # save model
    with open('xgb_spose_to_clip_cvbestmodel.pkl', 'wb') as f:
        pickle.dump(modelfull,f)


    random_cv.best_estimator_.save_model("best_xgb_spose_to_clip.txt")   # TODO: fix

    print( "Feature importance", random_cv.feature_importances_)  # TODO: save


    # TODO: turn off regularization? 
