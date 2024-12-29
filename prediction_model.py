from sklearn import tree
from matplotlib import pyplot as plot
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score

from main import *


def train_dt(model, encoded_data:pd.DataFrame):
    model.fit(encoded_data.iloc[:, :-1], encoded_data.iloc[:, -1:])
    return model

def predict(model:tree.DecisionTreeClassifier, encoded_data:pd.DataFrame):
    predictions = model.predict(encoded_data.iloc[:, :-1])
    return predictions

def print_metrics(predictions, gold_standard):
    print('accuracy=%f' % (accuracy_score(gold_standard.iloc[:, -1:], predictions)))
    print('precision=%f' % (precision_score(gold_standard.iloc[:, -1:], predictions, average='macro')))
    print('recall=%f' % (recall_score(gold_standard.iloc[:, -1:], predictions, average='macro')))
    print('f-measure=%f' % (f1_score(gold_standard.iloc[:, -1:], predictions, average='macro')))

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

    conc_cases = False
    avg_dur = True
    my_int1 = True
    my_int2 = True

    # Perform the encoding af the training set
    encoded_training_set = simple_index_encode(training_set, PREFIX_LENGTH, label_encoder, conc_cases=conc_cases,
                                               avg_dur=avg_dur, my_int1=my_int1, my_int2=my_int1)

    # Define the decision tree with the preferred parameters
    model = DecisionTreeClassifier(
        criterion='gini',
        # max_depth=9,
        # max_features=110,
        # min_samples_leaf=3,
        # min_samples_split=4,
        # max_leaf_nodes=47,
        # random_state=17493,
    )

    # Perform the training of the model
    model = train_dt(model, encoded_training_set)

    # Perform the encoding af the test set
    encoded_test_set = simple_index_encode(test_set, PREFIX_LENGTH, label_encoder, conc_cases=conc_cases,
                                           avg_dur=avg_dur, my_int1=my_int1, my_int2=my_int1)

    # Predict the outcome for the test set
    predictions = predict(model, encoded_test_set)

    # Evaluate the prediction's result
    print_metrics(predictions, encoded_test_set)

    # Plot the decision tree created during the training process
    # tree.plot_tree(model, fontsize=8)
    # plot.show()

    # model_optimization(encoded_training_set, max_evals=50000)
