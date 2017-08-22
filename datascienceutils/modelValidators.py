
# Add tests of independence, when a model predicts multi-class labels and probabilities
# For multi-class labels prediction, also test cross-correlation between the label frequencies

# When a model predicts binary classification, add tests of similarity to binomial distribution(with
# random prior or a given prior). Also check V-C dimension(https://datascience.stackexchange.com/questions/16140/how-to-calculate-vc-dimension/16146)

# When a model is unsupervised clustering, do tests of independence on the clusters
# generated/predicted

# Model is predicting probabilities, test the distribution (of the output) matches what is
# natural/common for the domain.

# When a model is predicting numeric/floating point values, check the distribution (of output) makes
# sense given the distribution (of inputs) in the bayesian sense(i.e: output dist and input dists
# make a conjugate pairs([https://www.johndcook.com/blog/conjugate_prior_diagram/] (a diagram here))
