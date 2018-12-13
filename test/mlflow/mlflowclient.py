
import string
import json
from mlflow import log_metric, log_param, log_artifact
import mlflow
import logging
import os

class AutoSkLearnMLFlowLogIntegrator:

    def __init__(self, experiment_name: string, tmp_log_path: string, score_metric : dict):
        self.tmp_log_path = tmp_log_path
        self.experiment_name = experiment_name
        self.score_metric = score_metric
        self.run_history_path = self.tmp_log_path + '/smac3-output'
        self.run_stats_path = self.tmp_log_path + '/smac3-output/run_1/stats.json'
        self.stub_for_run_histpry_file = '/runhistory.json'
        self.logger = logging.getLogger()
        return

    def log_param_dictionary(self, key_value_pairs):
        for key,value in key_value_pairs.items():
            if (type(value) is int) or (type(value) is float):
                log_metric(str(key).replace(':', "/"), value)
            else:
                log_param(str(key).replace(':', "/"), value)
        return


    def register_regressor_experiment(self):
        self.experiment_id = mlflow.create_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)
        for key,value in self.score_metric.items():
            log_metric(key,value)
        for dir_name, sub_dir_list, file_list in os.walk(self.run_history_path):
            for sub_dir in sub_dir_list:
                file_path_for_run = self.run_history_path + "/" + sub_dir + self.stub_for_run_histpry_file
                with open(file_path_for_run) as f:
                    run_data = json.load(f)
                    for i in range(0, len(run_data["data"])):
                        try:
                            log_metric("cost",run_data["data"][i][1][0])
                            log_metric("time", run_data["data"][i][1][1])
                            if "duration" in run_data["data"][i][1][3]:
                                log_metric("duration", run_data["data"][i][1][3]["duration"])
                            if "train_loss" in run_data["data"][i][1][3]:
                                log_metric("train_loss", run_data["data"][i][1][3]["train_loss"])
                            if "configuration_origin" in run_data["data"][i][1][3]:
                                log_param("configuration_origin", run_data["data"][i][1][3]["configuration_origin"])
                            self.log_param_dictionary(run_data["configs"][str(i+1)])
                        except Exception as e:
                            self.logger.error(e)
                            mlflow.end_run()
                            continue
                        mlflow.end_run(status="FINISHED")
        return





