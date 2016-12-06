from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score


def scores(y_test, y_predict, methods=None):
    score_methods = {
        "accuracy_score": [accuracy_score, {}],
        "precision:": [precision_score, {"average": "weighted"}],
        "recall:": [recall_score, {"average": "weighted"}],
        "f1_score:": [f1_score, {"average": "weighted"}]
    }
    if methods is None:
        methods = score_methods.keys()
    scores = {}
    for item in methods:
        method, args = score_methods[item]
        score = method(y_test, y_predict, **args)
        if item not in scores:
            scores[item] = [score]
        else:
            scores[item].append(score)
    return scores
