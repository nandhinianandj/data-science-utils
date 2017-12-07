import json
from . import statsutils as su

def validate(model, model_info_or_file, input_data):
    # when it's a naive bayes model check, conjugate pairs([https://www.johndcook.com/blog/conjugate_prior_diagram/] (a diagram here))
    if not isinstance(model_info_or_file, dict):
        with open(model_info, 'r') as fd:
            model_info = json.load(fd)
    else:
        model_info = model_info_or_file

    assert 'input_metadata' in model_info, "Input metadata missing in model info"
    assert 'output_metadata' in model_info, "Output metadata missing in model info"
    assert 'output_type' in model_info , "Output type required"
    assert 'model_class' in model_info, "Model Class (multi-class/single-class/regression) required"

    print('Input dist tests')
    for col in input_data.columns:
        dist = model_info['input_metadata'][col]['dist']
        su.distribution_similarity(input_data[col].tolist(), dist_type=dist)

    predictions = model.predict(input_data)
    series = predictions
    if model_info.get('model_class', None) == 'regression':
        if model_info.get('output_metadata', None):
            dist = model_info['output_metadata']['dist']
        else:
            dist = 'norm'
        print('Output dist tests')
        # Run k-s test for similarity between output and distribution
        test_results = su.distribution_similarity(series, dist)
        print(test_results)

    if model_info.get('model_class', None) == 'multiclass':
        # Check the predictions are type of categorical variables (float or int)
        print('Output dist tests')
        # Check for belongingness to a multinomial distribution.
        n_classes = model_info['output_metadata']['n_classes']
        class_probs = model_info['output_metadata']['class_probs']
        assert n_classes == len(class_probs), 'n_classes must equal length of class_probs'
        dist = stats.multinomial(len(input_data), class_probs)
        test_results = su.distribution_similarity(series, dist)

    if model_info.get('model_class', None) == 'singleclass':
        # When a model predicts binary classification, add tests of similarity to binomial distribution(with
        # random prior or a given prior). Also check V-C dimension(https://datascience.stackexchange.com/questions/16140/how-to-calculate-vc-dimension/16146)
        # parse and translate output_metadata to choice of tests
        print('Output dist tests')
        prob = model_info['output_metadata']['success_prob']
        # Check for belongingness to a binomial distribution.
        dist = stats.binom(len(input_data), prob ) #'binom'
        test_results = su.distribution_similarity(series, dist)

    if model_info.get('input_metadata', None):
        input_dists = model_info['input_metadata']
        for col, dist in input_dists.items():
            series = input_data[col].tolist()
            dist = dist['dist']
            test_results = su.distribution_similarity(series, dist)
    print(test_results)

# TODOS:
# When a model is unsupervised clustering, do tests of independence on the clusters
# generated/predicted

# Model is predicting probabilities, test the distribution (of the output) matches what is
# natural/common for the domain.

