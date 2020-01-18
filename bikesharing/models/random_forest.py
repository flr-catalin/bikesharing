import os
import pickle
import random
from pathlib import Path

import numpy
from matplotlib import pyplot
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import RandomizedSearchCV, KFold

from bikesharing.utilities.loader import Loader


class RandomForest:
    """
    RandomForestRegressor wrapper for the bike sharing dataset.
    """

    def __init__(self, csv_path):
        """
        Initialises the random forest wrapper.
        @param csv_path: the path to the .csv file
        @type csv_path: Path
        """
        loader = Loader(csv_path)
        random.seed(5)

        self.csv_data = {"full": loader.get_data()}
        self.target = ["cnt"]
        self.features = ["season", "mnth", "hr", "holiday", "weekday", "workingday",
                         "weathersit", "temp", "atemp", "hum", "windspeed"]

        self.samples = {"full": self.csv_data["full"][self.features].values}
        self.labels = {"full": self.csv_data["full"][self.target].values.ravel()}

        self.model = RandomForestRegressor(bootstrap=True, criterion="mse", max_depth=None,
                                           max_features="auto", max_leaf_nodes=None,
                                           min_impurity_decrease=0.0, min_impurity_split=None,
                                           min_samples_leaf=1, min_samples_split=4,
                                           min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
                                           oob_score=False, random_state=100, verbose=0, warm_start=False)

    def save_model(self, save_path):
        """
        Saves the model to a local file.
        @param save_path: The save path
        @type save_path: Path
        @return:
            True if the model was successfully saved
            False if the model is not initialised
        @rtype: bool
        """
        if self.model is not None:
            pickle.dump(self.model, open(str(save_path), "wb"))
            return True
        else:
            return False

    def load_model(self, load_path):
        """
        Loads the model from a local file.
        @param load_path: the load path
        @type load_path: Path
        @return:
            True if the model was successfully loaded
            False if the model file does not exist
        @rtype: bool
        """
        if os.path.exists(str(load_path)):
            self.model = pickle.load(open(str(load_path), "rb"))
            return True
        else:
            return False

    def random_parameter_search(self, iterations=100):
        """
        Configures the random grid's number of trees, number of features to consider at every split,
        the tree's maximum number of levels, the minimum number of samples needed to split a node
        the minimum number of samples needed at each leaf node and the method of selecting samples for
        each of the tree's training. With the random grid, performs a random search of parameters using
        3-fold cross-validation over 100 combinations.
        @param iterations: the number of iterations
        @type iterations: int
        @return: the best hyperparameters found
        @rtype: dict
        """
        estimators_number = [int(x) for x in numpy.linspace(start=10, stop=1000, num=10)]
        max_features = ["auto", "sqrt"]
        max_depth = [int(x) for x in numpy.linspace(10, 110, num=11)]
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]

        random_grid = {"estimators_number": estimators_number,
                       "max_features": max_features,
                       "max_depth": max_depth,
                       "min_samples_split": min_samples_split,
                       "min_samples_leaf": min_samples_leaf,
                       "bootstrap": bootstrap}

        random_forest = RandomForestRegressor()
        self.model = RandomizedSearchCV(estimator=random_forest, param_distributions=random_grid, n_iter=iterations,
                                        cv=3, verbose=2, random_state=0, n_jobs=-1)

        return self.model.get_params()

    def train(self, samples, labels):
        """
        Trains the model on the training data.
        @param samples: test samples
        @type samples: DataFrame
        @param labels: test labels
        @type labels: list
        """
        assert (len(samples) == len(labels))
        self.model.fit(samples, labels)

    def test(self, samples, labels):
        """
        Evaluates the model's performance on the test data.
        @param samples: test samples
        @type samples: DataFrame
        @param labels: test labels
        @type labels: list(int)
        @return the test results
        @rtype dict
        """
        if self.model is None:
            print("Please load or train a model before!")
            return

        assert (len(samples) == len(labels))

        prediction = self.model.predict(samples)

        score = self.model.score(samples, labels)
        mse = mean_squared_error(labels, prediction)
        mae = mean_absolute_error(labels, prediction)
        rmsle = numpy.sqrt(mean_squared_log_error(labels, prediction))

        return {"mse": mse, "mae": mae, "score": score, "rmsle": rmsle}

    def select_data(self, index_list):
        """
        Filters the full dataset depending on the indices in the provided index list.
        @param index_list: the list of sample indices
        @type index_list: list
        @return samples, labels: the samples and labels with the provided indices
        @rtype: list
        """
        samples = [self.samples["full"][i] for i in index_list]
        labels = [self.labels["full"][i] for i in index_list]

        return samples, labels

    def k_fold_cross_validation(self):
        """
        Runs a 3-fold cross-validation on the bike sharing dataset.
        """

        table = PrettyTable()
        table.field_names = ["Model", "Split", "Mean Squared Error", "Mean Absolute Error",
                             "SQRT of Mean Squared Log Error", "R² Score"]

        k_fold = KFold(n_splits=3, shuffle=True, random_state=100)

        res = []
        split = 1

        for train_index, test_index in k_fold.split(self.samples["full"]):
            samples, labels = {}, {}
            samples["train"], labels["train"] = self.select_data(train_index)
            samples["test"], labels["test"] = self.select_data(test_index)

            self.train(samples["train"], labels["train"])
            res.append(self.test(samples["test"], labels["test"]))
            table.add_row(
                [type(self.model).__name__, split, format(res[-1]["mse"], ".2f"), format(res[-1]["mae"], ".2f"),
                 format(res[-1]["rmsle"], ".2f"), format(res[-1]["score"], ".2f")])
            split += 1

        score = numpy.mean([item["score"] for item in res])
        mse = numpy.mean([item["mse"] for item in res])
        mae = numpy.mean([item["mae"] for item in res])
        rmsle = numpy.mean([item["rmsle"] for item in res])

        table.add_row([type(self.model).__name__, "Mean", format(mse, ".2f"), format(mae, ".2f"), format(rmsle, ".2f"),
                       format(score, ".2f")])

        print(table)

    def feature_importance(self):
        """
        Sets first split as default training data, gets the sorted feature importance,
        prints it and plots it.
        """
        if self.model is None:
            print("Please load or train a model before!")
            return

        kf = KFold(n_splits=3, shuffle=True, random_state=100)
        train_index, test_index = next(kf.split(self.samples["full"]))
        samples, labels = self.select_data(train_index)

        self.model.fit(samples, labels)

        importance = self.model.feature_importances_
        std = numpy.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        indices = numpy.argsort(importance)[::-1]

        print("Feature ranking:")
        for f in range(len(self.features)):
            print("%d. feature %s (%f)" % (f + 1, self.features[indices[f]], importance[indices[f]]))

        pyplot.figure()
        pyplot.title("Feature importance")
        pyplot.bar(range(len(self.features)), importance[indices], color="cyan", yerr=std[indices],
                   align="center")
        pyplot.xticks(range(len(self.features)), [self.features[i] for i in indices])
        pyplot.show()
