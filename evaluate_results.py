import re
import os
from collections import defaultdict

def average_log_by_combination(filename):

    metrics_pattern = r"Features: ([\w, ]+)\nResult: Accuracy = ([\d\.]+), Precision = ([\d\.]+), Recall = ([\d\.]+), F1-score = ([\d\.]+)"

    feature_metrics = defaultdict(lambda: {"Accuracy": [], "Precision": [], "Recall": [], "F1-score": []})

    with open(filename, "r") as f:
        log_content = f.read()

    log_blocks = log_content.split("-" * 119)

    for block in log_blocks:
        matches = re.findall(metrics_pattern, block)
        for match in matches:
            features, acc, prec, rec, f1 = match[0], float(match[1]), float(match[2]), float(match[3]), float(match[4])
            feature_metrics[features]["Accuracy"].append(acc)
            feature_metrics[features]["Precision"].append(prec)
            feature_metrics[features]["Recall"].append(rec)
            feature_metrics[features]["F1-score"].append(f1)

    print("Average metrics by feature combination:")
    for features, metrics in feature_metrics.items():
        avg_acc = sum(metrics["Accuracy"]) / len(metrics["Accuracy"])
        avg_prec = sum(metrics["Precision"]) / len(metrics["Precision"])
        avg_rec = sum(metrics["Recall"]) / len(metrics["Recall"])
        avg_f1 = sum(metrics["F1-score"]) / len(metrics["F1-score"])
        print(f"Features: {features}")
        print(f"  Accuracy: {avg_acc:.4f}, Precision: {avg_prec:.4f}, Recall: {avg_rec:.4f}, F1-score: {avg_f1:.4f}")
        # print(f"{avg_acc:.4f}\t{avg_prec:.4f}\t{avg_rec:.4f}\t{avg_f1:.4f}")


def average_log(filename):

    metrics_pattern = r"Accuracy = ([\d\.]+), Precision = ([\d\.]+), Recall = ([\d\.]+), F1-score = ([\d\.]+)"
    best_result_pattern = r"BEST Result: Accuracy = ([\d\.]+), Precision = ([\d\.]+), Recall = ([\d\.]+), F1-score = ([\d\.]+)"

    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []

    best_accuracies = []
    best_precisions = []
    best_recalls = []
    best_f1_scores = []

    with open(filename, "r") as f:
        log_content = f.read()

    log_blocks = log_content.split("-" * 119)

    for block in log_blocks:
        matches = re.findall(metrics_pattern, block)
        best_match = re.search(best_result_pattern, block)

        if matches:
            for match in matches:
                acc, prec, rec, f1 = map(float, match)
                all_accuracies.append(acc)
                all_precisions.append(prec)
                all_recalls.append(rec)
                all_f1_scores.append(f1)

        if best_match:
            best_acc, best_prec, best_rec, best_f1 = map(float, best_match.groups())
            best_accuracies.append(best_acc)
            best_precisions.append(best_prec)
            best_recalls.append(best_rec)
            best_f1_scores.append(best_f1)

    average_accuracies = sum(all_accuracies) / len(all_accuracies)
    average_precisions = sum(all_precisions) / len(all_precisions)
    average_recalls = sum(all_recalls) / len(all_recalls)
    average_f1_scores = sum(all_f1_scores) / len(all_f1_scores)

    average_best_accuracies = sum(best_accuracies) / len(best_accuracies)
    average_best_precisions = sum(best_precisions) / len(best_precisions)
    average_best_recalls = sum(best_recalls) / len(best_recalls)
    average_best_f1_scores = sum(best_f1_scores) / len(best_f1_scores)

    print("Average metrics: ")
    print(f"Accuracy: {average_accuracies:.4f}")
    print(f"Precision: {average_precisions:.4f}")
    print(f"Recall: {average_recalls:.4f}")
    print(f"F1-score: {average_f1_scores:.4f}\n")

    print("Average BEST Result:")
    print(f"Accuracy: {average_best_accuracies:.4f}")
    print(f"Precision: {average_best_precisions:.4f}")
    print(f"Recall: {average_best_recalls:.4f}")
    print(f"F1-score: {average_best_f1_scores:.4f}\n")


if __name__ == '__main__':

    # for i in range(25):
    #     print(f"Iteration #{i}")
    #     os.system("python prediction_model.py")

    average_log("./results.log")
    average_log_by_combination("./results.log")