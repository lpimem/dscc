from sklearn.svm import SVC

from pprint import pprint

from .evaluate import scores
from .dataset import DataSet, load_news_group_ds
from .cluster import label_by_cluster, map_labels
from . import cache


def train_classifier(X, Y, model):
    model.fit(X, Y)
    return model


def exp(ds, cluster_method, classier, n_clusters, dr_method=None, use_cache=True):
    exp_name = "{ds.name}_{cluster_method}_{n_clusters}_{dr_method}"\
        .format(**locals())
    print(" =" * 4, exp_name, "= " * 4)
    # reduce dimention
    if dr_method is not None:
        ds.reduce_dim(dr_method)
    # regularization
    ds.regularize()
    # load from cache
    if use_cache:
        labels = cache.load(exp_name)
    else:
        labels = None
    # clustering
    if labels is None:
        labels = label_by_cluster(ds.X, cluster_method, n_clusters)
        cache.save(exp_name, labels)
    # map test data labels to clustered labels for evaluation
    labels_test = map_labels(ds.Y, labels, ds.Y_test)
    # train classification models
    model = train_classifier(ds.X, labels, classier)
    pred = model.predict(ds.X_test)
    s = scores(labels_test, pred, ["accuracy_score"])
    print("-" * 10)
    print(exp_name)
    pprint(s)
    print("-" * 10)
    return s["accuracy_score"][0]


def classier_only(ds, classier):
    # regularize
    ds.regularize()
    # classification
    model = train_classifier(ds.X, ds.Y, classier)
    pred = model.predict(ds.X_test)
    print('-' * 10)
    print("Classifier only:")
    s = scores(ds.Y_test, pred, ["accuracy_score"])
    pprint(s)
    print('-' * 10)
    return s["accuracy_score"][0]


def init_svc():
    return SVC(kernel="rbf", tol=0.001, decision_function_shape="ovr")

def assemble_result_df(cmethods, dr_methods, scores_acc):
    import pandas as pd
    df = {}
    n = len(cmethods)
    return pd.DataFrame({
        "model": pd.Series(cmethods),
        "dim reduction": pd.Series(dr_methods),
        "accuracy": pd.Series(scores_acc)
    })

def plot(df):
    print("-"*10)
    print(df)
    print("-"*10)
    from bokeh.charts import Bar, show
    p = Bar(df, label=["model", "dim reduction"], \
            values="accuracy", agg="mean", color="model",\
            legend=None, bar_width=0.3, plot_width=600, plot_height=600)
    p.logo = None
    # will open a browser to show the plot.
    show(p)


def main(use_cache=True):
    df = cache.load("result") if use_cache else None
    if df is None:
        N_CATS = 2
        ds = load_news_group_ds(N_CATS)
        clustering_methods = [
            "KMeans",
            "SpectralClustering",
            "AgglomerativeClustering",
            "FuzzyKMeans",
        ]

        exp_r = 0
        exp_r_pre = 0
        cls_r = 0
        cls_r_pre = 0
        cmethods = []
        dr_methods = []
        scores_acc = []
        def fill_result(cm, dm, r, cr):
            for i in [0, 1]:
                cmethods.append([cm+"+SVM", "SVM"][i])
                dr_methods.append(dm)
                scores_acc.append([r, cr][i])

        for cmethod in clustering_methods:
            for dr_method in DataSet.DIM_REDUCTION_METHODS:
                try:
                    exp_r = exp(ds, cmethod, init_svc(), N_CATS,
                        dr_method=dr_method, use_cache=True)
                except:
                    exp_r = exp_r_pre
                finally:
                    exp_r_pre = exp_r
                
                try: 
                    cls_r = classier_only(ds, init_svc())
                except:
                    cls_r = cls_r_pre
                finally:
                    cls_r_pre = cls_r
                
                fill_result(cmethod, dr_method, exp_r, cls_r)
                ds.restore()
        # pack result into a pandas DataFrame object for plotting
        df = assemble_result_df(cmethods, dr_methods, scores_acc)
        # save packed result for future use.
        cache.save("result", df)
    plot(df)

if __name__ == '__main__':
    main(use_cache=True)
