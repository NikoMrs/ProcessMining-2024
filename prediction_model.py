from sklearn import tree
from matplotlib import pyplot as plot
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score

from itertools import product
from datetime import datetime

import random

from encoding import *


def train_dt(model, encoded_data:pd.DataFrame):
    model.fit(encoded_data.iloc[:, :-1], encoded_data.iloc[:, -1:])
    return model

def predict(model:tree.DecisionTreeClassifier, encoded_data:pd.DataFrame):
    predictions = model.predict(encoded_data.iloc[:, :-1])
    return predictions

def get_metrics(predictions, gold_standard):
    metrics = {
        'accuracy': accuracy_score(gold_standard.iloc[:, -1:], predictions),
        'precision': precision_score(gold_standard.iloc[:, -1:], predictions, average='macro', zero_division=0),
        'recall': recall_score(gold_standard.iloc[:, -1:], predictions, average='macro', zero_division=0),
        'f1_score': f1_score(gold_standard.iloc[:, -1:], predictions, average='macro', zero_division=0)
    }

    return metrics

def model_optimization(encoded_data:pd.DataFrame, max_evals=1000):

    param_space = {
        'max_depth': hp.choice('max_depth', range(1, 30)),
        'max_features': hp.choice('max_features', range(1, 200)),
        'min_samples_split': hp.choice('min_samples_split', range(2, 100)),
        'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 100)),
        'max_leaf_nodes': hp.choice('max_leaf_nodes', range(2, 100)),
        'criterion': hp.choice('criterion', ["gini", "entropy", "log_loss"])
    }

    def acc_model(params):
        X_ = encoded_data.iloc[:, :-1]
        y = encoded_data.iloc[:, -1:]
        model = tree.DecisionTreeClassifier(**params)
        return cross_val_score(model, X_, y, scoring='accuracy').mean()

    def f(params):
        acc = acc_model(params)
        return {'loss': -acc, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(f, param_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    print(best)
    print(best['criterion'], best['criterion'])
    return best, trials

if __name__ == '__main__':
    PREFIX_LENGTH = 5

    training_set = import_xes("./Production_avg_dur_training_0-80.xes")
    test_set = import_xes("./Production_avg_dur_testing_80-100.xes")

    # Compute the label_encoder looking at the activity of the training set
    label_encoder = get_label_encoder(training_set)

    feature_names = ["Concurrent_cases", "Average_duration", "Average_resources", "Overlapping_duration",
                     "Concurrent_cases_per_event"]
    combinations = list(product([False, True], repeat=len(feature_names)))

    STATE_SEED = 123456
    STATE_SEED = random.randint(0, 4294967295)
    current_time = datetime.now()

    header_string = f"{current_time.strftime('%d/%m/%Y %H:%M:%S')} - SEED {STATE_SEED}"
    padding_length = (119 - len(header_string) - 2) // 2
    line = f"{'-' * padding_length} {header_string} {'-' * padding_length}"
    line = line if len(line) == 119 else line + '-'

    best_results = None
    best_combination = None
    with open("./results.log", "a") as f:

        f.write(f"{line}\n\n\n")

        for combination in combinations:
            print(combination)
            conc_cases, avg_dur, my_int1, my_int2, my_int3 = combination

            # Training
            encoded_train = simple_index_encode(training_set, PREFIX_LENGTH, label_encoder, conc_cases, avg_dur,
                                                my_int1, my_int2, my_int3)

            model = DecisionTreeClassifier(random_state=STATE_SEED)
            trained_model = train_dt(model, encoded_train)

            # Testing
            encoded_test = simple_index_encode(test_set, PREFIX_LENGTH, label_encoder, conc_cases, avg_dur,
                                               my_int1, my_int2, my_int3)

            predictions = predict(trained_model, encoded_test)
            metrics = get_metrics(predictions, encoded_test)

            features_used = [feature for feature, included in zip(feature_names, combination) if included]
            features_description = "Features: " + ", ".join(features_used) if features_used else "Features: None"

            # Write results on file
            f.write(f"{features_description}\n")
            f.write(f"Result: Accuracy = {metrics['accuracy']:.4f}, Precision = {metrics['precision']:.4f}, "
                    f"Recall = {metrics['recall']:.4f}, F1-score = {metrics['f1_score']:.4f}\n\n")

            if best_combination is None:
                best_combination = combination
                best_results = metrics
            elif metrics['precision'] >= best_results['precision']:
                best_combination = combination
                best_results = metrics

        features_used = [feature for feature, included in zip(feature_names, best_combination) if included]
        features_description = ", ".join(features_used) if features_used else "Features: None"
        f.write(f"BEST Combination: {features_description}\n")
        f.write(f"BEST Result: Accuracy = {best_results['accuracy']:.4f}, Precision = {best_results['precision']:.4f}, "
                f"Recall = {best_results['recall']:.4f}, F1-score = {best_results['f1_score']:.4f}\n\n")
        f.write(f"{'-' * 119}\n\n")

    # conc_cases = False
    # avg_dur = False
    # my_int1 = False
    # my_int2 = False
    # my_int3 = True
    #
    # # Perform the encoding af the training set
    # encoded_training_set = simple_index_encode(training_set, PREFIX_LENGTH, label_encoder, conc_cases=conc_cases,
    #                                            avg_dur=avg_dur, my_int1=my_int1, my_int2=my_int2, my_int3=my_int3)
    #
    # # Define the decision tree with the preferred parameters
    # model = DecisionTreeClassifier(
    #     criterion='gini',
    #     # max_depth=9,
    #     # max_features=110,
    #     # min_samples_leaf=3,
    #     # min_samples_split=4,
    #     # max_leaf_nodes=47,
    #     # random_state=17493,
    # )
    #
    # # Perform the training of the model
    # model = train_dt(model, encoded_training_set)
    #
    # # Perform the encoding af the test set
    # encoded_test_set = simple_index_encode(test_set, PREFIX_LENGTH, label_encoder, conc_cases=conc_cases,
    #                                        avg_dur=avg_dur, my_int1=my_int1, my_int2=my_int2, my_int3=my_int3)
    #
    # # Predict the outcome for the test set
    # predictions = predict(model, encoded_test_set)
    #
    # # Evaluate the prediction's result
    # print_metrics(predictions, encoded_test_set)
    #
    # # Plot the decision tree created during the training process
    # # tree.plot_tree(model, fontsize=8)
    # # plot.show()
    #
    # # model_optimization(encoded_training_set, max_evals=50000)
