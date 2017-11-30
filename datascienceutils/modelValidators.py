import json

def validate(model, model_info_file, input_data):
    with open(model_info, 'r') as fd:
        model_info = json.load(fd)
    assert 'input_metadata' in model_info, "Input metadata missing in model info"
    assert 'output_metadata' in model_info, "Output metadata missing in model info"
    assert 'output_type' in model_info , "Output type required"
    assert 'model_class' in model_info, "Model Class (multi-class/single-class/regression) required"
    if model_info.output_metadata:
        pass
    if model_info.model_class == 'regression':
        predictions = model.predict(input_data)
        # Check the predictions are type of continuous variables (float or int)
        # parse and translate output_metadata to choice of tests
        pass
    if model_info.model_class == 'multiclass':
        # Check the predictions are type of categorical variables (float or int)
        # parse and translate output_metadata to choice of tests
        pass
    if model_info.input_metadata:
        input_cols = model_info.input_metadata['ncols']
        input_dists = model_info.input_metadata['input_dists']
        check_dists()

        pass

    pass
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
