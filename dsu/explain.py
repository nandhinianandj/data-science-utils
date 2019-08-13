def interpret(dataframe, model):
    features = dataframe.columns
    from skater import Interpretation
    from skater.model import InMemoryModel
    df_np = dataframe.as_matrix()
    interpreter = Interpretation(df_np, feature_names=features)
    skater_model = InMemoryModel(model.predict, examples=df_np[:10])
    interpreter.feature_importance.feature_importance(skater_model)
    interpreter.partial_dependence.plot_partial_dependence([features[0], features[1]], skater_model)
