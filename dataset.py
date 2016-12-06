
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


class DataSet:
    DIM_REDUCTION_METHODS = ["pca", "select_k_best"]

    def __init__(self, name, X, Y, X_test, Y_test):
        self.name = name
        self.origin_X, self.origin_Y, self.origin_X_test, self.origin_Y_test \
            = (X, Y, X_test, Y_test)
        self.X, self.Y, self.X_test, self.Y_test = (X, Y, X_test, Y_test)
        self._dr = False
        self._r = False

    def regularize(self):
        if self._r:
            return
        ds = self
        scaler = StandardScaler(with_mean=False).fit(ds.X)
        self.X = scaler.transform(ds.X)
        self.X_test = scaler.transform(ds.X_test)
        self._r = True

    def reduce_dim(self, method):
        if self._dr:
            return
        if method == "pca":
            self.apply_pca(n_c=10)
        elif method == "select_k_best":
            self.apply_select_k_best()
        else:
            print(method, "method not found. Dataset is not changed")
        if method in DataSet.DIM_REDUCTION_METHODS:
            self._dr = True

    def restore(self):
        self.X, self.Y, self.X_test, self.Y_test = \
            self.origin_X, self.origin_Y, \
            self.origin_X_test, self.origin_Y_test
        self._r = False
        self._dr = False

    def apply_pca(self, n_c=None):
        args = {}
        if n_c is None:
            if self.X.shape[0] < self.X[0].shape[0]:
                # will reduce dimentions to [m, m] 
                #     where m=min(n_samples, n_features)
                args = {
                    "n_components": "mle",
                    "svd_solver": 'full'
                }
        else:
            args = {
                "n_components": n_c
            }
        pca = PCA(**args)
        self.X = pca.fit_transform(self.origin_X.toarray(), self.origin_Y)
        self.X_test = pca.transform(self.origin_X_test.toarray())

    def apply_select_k_best(self, k=10):
        ch2 = SelectKBest(chi2, k=k)
        self.X = ch2.fit_transform(self.origin_X, self.origin_Y)
        self.X_test = ch2.transform(self.origin_X_test)


def load_news_group_ds(n_cats):
    cats = [
        'comp.graphics',
        'rec.autos',
        'talk.politics.guns',
        'sci.med',
        'rec.sport.baseball',
    ][:n_cats]
    remove = ('headers', 'footers', 'quotes')
    newsgroups_train = fetch_20newsgroups(subset='train',
                                          categories=cats,
                                          shuffle=True,
                                          random_state=42,
                                          remove=remove)
    newsgroups_test = fetch_20newsgroups(subset='test',
                                         categories=cats,
                                         shuffle=True,
                                         random_state=42,
                                         remove=remove)
    # ---------
    # Taken from sk-learn website. 
    # Print loaded data size .
    def size_mb(docs):
        return sum(len(s.encode('utf-8')) for s in docs) / 1e6

    data_train_size_mb = size_mb(newsgroups_train.data)
    data_test_size_mb = size_mb(newsgroups_test.data)
    print("%d documents - %0.3fMB (training set)" % (
        len(newsgroups_train.data), data_train_size_mb))
    print("%d documents - %0.3fMB (test set)" % (
        len(newsgroups_test.data), data_test_size_mb))
    print("%d categories" % n_cats)
    print()
    # ---------

    vectorizor = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X = vectorizor.fit_transform(newsgroups_train.data)
    Y = newsgroups_train.target
    X_test = vectorizor.transform(newsgroups_test.data)
    Y_test = newsgroups_test.target
    return DataSet("d20newsgroups", X, Y, X_test, Y_test)
