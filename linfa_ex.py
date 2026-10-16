# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

# * File Name : linfa_ex.py
#
# * Purpose :
#
# * Creation Date : 03-12-2019
#
# * Last Modified : Tuesday 03 December 2019 10:19:16 PM IST
#
# * Created By :

# _._._._._._._._._._._._._._._._._._._._._.#

## From [here.](https://www.lpalmieri.com/posts/2019-12-01-taking-ml-to-production-with-rust-a-25x-speedup/)

from linfa import KMeans
from sklearn.datasets import make_blobs
import pytest
from sklearn.cluster import KMeans as sk_KMeans


@pytest.fixture(scope="session", autouse=True)
def make_data():
    return make_blobs(n_samples=1000000)


def test_k_means_rust(benchmark, make_data):
    dataset, cluster_index = make_data
    model = KMeans(3, max_iter=100, tol=1e-4)
    labels = benchmark(model.fit_predict, dataset)
    assert len(labels) == len(cluster_index)


def test_k_means_python(benchmark, make_data):
    dataset, cluster_index = make_data
    # Using the same algorithm
    model = sk_KMeans(
        3, init="random", algorithm="full", max_iter=100, tol=1e-4, n_init=1
    )
    labels = benchmark(model.fit_predict, dataset)
    assert len(labels) == len(cluster_index)
