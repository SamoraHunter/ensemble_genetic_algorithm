def test_plot_initial_feature_importance_type_error_from_none_values():
    """Test plot_initial_feature_importance TypeError exception handling.

    This test covers lines 730-738 in GA_results_explorer.py where:
    - Lines 730-734: Warning logged when no features found after decoding
    - Lines 735-739: TypeError exception handling when processing f_list

    The TypeError can occur during pd.Series creation from decoded feature lists,
    for example when the list contains None values that cannot be iterated.

    This test specifically targets line 735 (except TypeError) by providing
    malformed data that causes a TypeError during Series construction at lines 703-706.
    """
    import json

    import pandas as pd

    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": ["[[(0.5, 'Model', [1, 0, 1], 0, 0.9, None)]]"],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
            "auc": [0.85],
            "f_list": [["[1, 0, 1]"]],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    # Manually inject malformed data into feature_names to trigger TypeError
    # at the Series creation step (lines 703-706)
    explorer.df["feature_names"] = [
        [None]  # None can't be iterated in inner loop, causing TypeError
    ]

    result = explorer.plot_initial_feature_importance(outcome_variable="auc")

    assert result is None
