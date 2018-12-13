import datetime
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.regression
from mlflow import log_metric, log_param, log_artifact
from test.mlflow.mlflowclient import AutoSkLearnMLFlowLogIntegrator

def main():
    X, y = sklearn.datasets.load_boston(return_X_y=True)
    feature_types = (['numerical'] * 3) + ['categorical'] + (['numerical'] * 9)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    unique_id = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    tmp_folder_path = '/Users/kalyanan/Desktop/auto-sklearn-mlflow/regression_example_tmp-' + unique_id
    output_folder_path = '/Users/kalyanan/Desktop/auto-sklearn-mlflow/regression_example_out-' + unique_id

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=480,
        per_run_time_limit=30,
        tmp_folder=tmp_folder_path,
        output_folder=output_folder_path,
        delete_tmp_folder_after_terminate=False,
        delete_output_folder_after_terminate=False
    )
    automl.fit(X_train, y_train, dataset_name='boston',
               feat_type=feature_types)
    print(automl.show_models())
    predictions = automl.predict(X_test)
    print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))
    log_metric("R2_score", sklearn.metrics.r2_score(y_test, predictions))
    mlflow_registry_client = AutoSkLearnMLFlowLogIntegrator(
        "boston-pricing-" + unique_id, tmp_folder_path,
        {"r2_score": sklearn.metrics.r2_score(y_test, predictions)})
    mlflow_registry_client.register_regressor_experiment()


if __name__ == '__main__':
    main()

