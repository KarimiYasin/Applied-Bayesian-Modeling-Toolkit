def get_confusion_matrix(true_y, pred_y):

    tp = fp = tn = fn = 0

    for actual, predicted in zip(true_y, pred_y):
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 1 and predicted == 0:
            fn += 1

    return tp, fp, tn, fn


def calculate_metrics(true_y, pred_y):

    tp, fp, tn, fn = get_confusion_matrix(true_y, pred_y)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if (precision + recall) == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score
    }

if __name__ == "__main__":

    true_labels = [1] * 10 + [0] * 90
    predicted_labels = [1] * 5 + [0] * 5 + [1] * 3 + [0] * 87

    results = calculate_metrics(true_labels, predicted_labels)

    print("--- Model Evaluation Results ---")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")