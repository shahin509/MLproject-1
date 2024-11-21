import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {

                "Random Forest": {
                    'n_estimators': [50, 100, 200, 300],  # Increased focus on realistic estimator counts
                    'max_depth': [None, 10, 20, 30, 50],  # Same as Decision Tree for depth
                    'min_samples_split': [2, 5, 10],  # Matches Decision Tree's split
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False],  # Bootstrap sampling for bagging
                },

                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],  # Removed 'poisson' as it's usually for count data
                    'splitter': ['best', 'random'],  # Useful for exploring different splitting strategies
                    'max_depth': [None, 10, 20, 30, 50],  # Limits to prevent overfitting
                    'min_samples_split': [2, 5, 10],  # Controls minimum samples to split a node
                    'min_samples_leaf': [1, 2, 4],  # Controls minimum samples for a leaf node
                },

                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Added slightly higher learning rates
                    'n_estimators': [50, 100, 200, 300],  # Consistent with Random Forest
                    'subsample': [0.7, 0.8, 0.9, 1.0],  # Slightly tighter range for subsampling
                    'max_depth': [3, 5, 7],  # Typical depths for Gradient Boosting
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                },

                "Linear Regression": {
                    # Linear regression usually doesn't have hyperparameters but can include normalization
                    'fit_intercept': [True, False],
                    'normalize': [True, False],  # Deprecated in some libraries but relevant in older versions
                },

                "XGBRegressor": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7],  # Controls tree complexity
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],  # Feature subsampling
                    'reg_alpha': [0, 0.1, 1],  # L1 regularization
                    'reg_lambda': [1, 1.5, 2],  # L2 regularization
                },

                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [100, 200, 300],  # Increased from lower values for more training iterations
                    'l2_leaf_reg': [3, 5, 7],  # Regularization parameter
                },

                "AdaBoost Regressor": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Removed overly small/large values
                    'n_estimators': [50, 100, 200, 300],
                    'loss': ['linear', 'square', 'exponential'],  # Commonly used loss functions
                }

            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[ list(model_report.values()).index(best_model_score) ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return best_model,r2_square
            
        except Exception as e:
            raise CustomException(e,sys)