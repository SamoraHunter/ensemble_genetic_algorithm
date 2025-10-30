import logging
from typing import Dict, List, Tuple

from fuzzysearch import find_near_matches

from ml_grid.pipeline.data_plot_split import (
    plot_candidate_feature_category_lists,
    plot_dict_values,
)
from ml_grid.util.global_params import global_parameters

logger = logging.getLogger("ensemble_ga")


def filter_substring_list(string_list: List[str], substr_list: List[str]) -> List[str]:
    """Filters a list of strings based on a list of substrings.

    Args:
        string_list: The list of strings to filter.
        substr_list: The list of substrings to search for.

    Returns:
        A new list containing strings from `string_list` that contain any of
        the substrings from `substr_list`, excluding any strings that contain "bmi".
    """
    return [
        s
        for s in string_list
        if any(sub in s for sub in substr_list) and "bmi" not in s
    ]


def get_pertubation_columns(
    all_df_columns: List[str],
    local_param_dict: Dict,
    drop_term_list: List[str],
) -> Tuple[List[str], List[str]]:
    """Categorizes and filters DataFrame columns for analysis.

    This function processes a list of column names from a clinical dataset,
    categorizing them by data type (e.g., blood tests, medications,
    demographics) based on substrings in their names. It returns two lists:
    one with columns selected for analysis (`pertubation_columns`) based on
    flags in `local_param_dict`, and another with columns to be dropped.

    The categorization logic is as follows:
    1.  Initial columns are dropped if they contain "__index_level", "Unnamed:",
        or "client_idcode:", or if they match terms in `drop_term_list`.
    2.  Columns are then grouped into categories like 'bloods', 'diagnostic_order',
        'drug_order', 'annotations', 'demographics', etc., based on predefined
        substrings.
    3.  A special post-processing step ensures that columns categorized as 'bloods'
        do not overlap with other categories to prevent ambiguity.
    4.  Based on boolean flags in `local_param_dict['data']`, columns from the
        enabled categories are collected into the final `pertubation_columns` list.
    5.  Logging and plotting of category counts can be enabled via the global
    flags in `local_param_dict`, and another with columns to be dropped.

    Args:
        all_df_columns: A list of all column names from the DataFrame.
        local_param_dict: A configuration dictionary containing:
            - 'data' (Dict): Boolean flags for each data category (e.g.,
              'age', 'sex', 'bloods').
            - 'outcome_var_n' (int): An identifier for the outcome variable.
        drop_term_list: A list of terms/substrings to identify columns that
            should be dropped from the analysis.

    Returns:
        A tuple containing:
            - pertubation_columns (List[str]): Column names selected for
              analysis based on the enabled categories in `local_param_dict`.
            - drop_list (List[str]): Column names to be dropped, including
              index levels, unnamed columns, and columns matching `drop_term_list`.
    """

    global_params = global_parameters()

    verbose = global_params.verbose

    orignal_feature_names = all_df_columns

    drop_list = []

    index_level_list = list(filter(lambda k: "__index_level" in k, all_df_columns))

    drop_list.extend(index_level_list)

    Unnamed_list = list(filter(lambda k: "Unnamed:" in k, all_df_columns))

    drop_list.extend(Unnamed_list)

    Unnamed_list = list(filter(lambda k: "client_idcode:" in k, all_df_columns))

    drop_list.extend(Unnamed_list)

    outcome_variable = f'outcome_var_{local_param_dict.get("outcome_var_n")}'

    for i in range(0, len(drop_term_list)):

        drop_term_string = drop_term_list[i]

        for elem in all_df_columns:
            res = find_near_matches(drop_term_string, elem.lower(), max_l_dist=0)

            if len(res) > 0:

                drop_list.append(elem)

    blood_test_substrings = [
        "_mean",
        "_median",
        "_mode",
        "_std",
        "_num-tests",
        "_days-since-last-test",
        "_max",
        "_min",
        "_most-recent",
        "_earliest-test",
        "_days-between-first-last",
        "_contains-extreme-low",
        "_contains-extreme-high",
    ]

    diagnostic_test_substrings = [
        "_num-diagnostic-order",
        "_days-since-last-diagnostic-order",
        "_days-between-first-last-diagnostic",
    ]

    drug_order_substrings = [
        "_num-drug-order",
        "_days-since-last-drug-order",
        "_days-between-first-last-drug",
    ]

    meta_sp_annotation_count_list = list(
        filter(lambda k: "_count_subject_present" in k, all_df_columns)
    )

    not_meta_sp_annotation_count_list = list(
        filter(lambda k: "_count_subject_not_present" in k, all_df_columns)
    )

    meta_rp_annotation_count_list = list(
        filter(lambda k: "_count_relative_present" in k, all_df_columns)
    )

    not_meta_rp_annotation_count_list = list(
        filter(lambda k: "_count_relative_not_present" in k, all_df_columns)
    )

    meta_sp_annotation_count_list.extend(not_meta_sp_annotation_count_list)

    meta_sp_annotation_count_list.extend(meta_rp_annotation_count_list)

    meta_sp_annotation_count_list.extend(not_meta_rp_annotation_count_list)

    annotation_count_list = list(
        filter(
            lambda k: "_count" in k and "_count_subject_present" not in k,
            all_df_columns,
        )
    )

    appointments_substrings = ["ConsultantCode_", "ClinicCode_", "AppointmentType_"]

    diagnostic_order_list = []
    diagnostic_list = filter_substring_list(all_df_columns, diagnostic_test_substrings)
    diagnostic_order_list.extend(diagnostic_list)

    drug_order_list = []
    drug_list = filter_substring_list(all_df_columns, drug_order_substrings)
    drug_order_list.extend(drug_list)

    appointments_list = []
    appointments = filter_substring_list(all_df_columns, appointments_substrings)
    appointments_list.extend(appointments)

    bmi_list = list(filter(lambda k: "bmi_" in k, all_df_columns))

    ethnicity_list = list(filter(lambda k: "census_" in k, all_df_columns))

    annotation_mrc_count_list = list(
        filter(lambda k: "_count_mrc_cs" in k, all_df_columns)
    )

    meta_sp_annotation_mrc_count_list = list(
        filter(lambda k: "_count_subject_present_mrc_cs" in k, all_df_columns)
    )

    not_meta_sp_annotation_mrc_count_list = list(
        filter(lambda k: "_count_subject_not_present_mrc_cs" in k, all_df_columns)
    )

    relative_meta_rp_annotation_mrc_count_list = list(
        filter(lambda k: "_count_relative_present_mrc_cs" in k, all_df_columns)
    )

    not_relative_meta_rp_annotation_mrc_count_list = list(
        filter(lambda k: "_count_relative_not_present_mrc_cs" in k, all_df_columns)
    )

    meta_sp_annotation_mrc_count_list.extend(not_meta_sp_annotation_mrc_count_list)

    meta_sp_annotation_mrc_count_list.extend(relative_meta_rp_annotation_mrc_count_list)

    meta_sp_annotation_mrc_count_list.extend(
        not_relative_meta_rp_annotation_mrc_count_list
    )

    core_02_list = list(filter(lambda k: "core_02_" in k, all_df_columns))

    bed_list = list(filter(lambda k: "bed_" in k, all_df_columns))

    vte_status_list = list(filter(lambda k: "vte_status_" in k, all_df_columns))

    hosp_site_list = list(filter(lambda k: "hosp_site_" in k, all_df_columns))

    core_resus_list = list(filter(lambda k: "core_resus_" in k, all_df_columns))

    news_list = list(filter(lambda k: "news_resus_" in k, all_df_columns))

    bloods_list = filter_substring_list(all_df_columns, blood_test_substrings)

    date_time_stamp_list = list(
        filter(lambda k: "date_time_stamp" in k, all_df_columns)
    )

    # Combine these into a single conceptual list for overlap check later
    meta_sp_annotation_all_counts = (
        meta_sp_annotation_count_list
        + not_meta_sp_annotation_count_list
        + meta_rp_annotation_count_list
        + not_meta_rp_annotation_count_list
    )
    # Combine these into a single conceptual list for overlap check later
    meta_sp_annotation_mrc_all_counts = (
        meta_sp_annotation_mrc_count_list
        + not_meta_sp_annotation_mrc_count_list
        + relative_meta_rp_annotation_mrc_count_list
        + not_relative_meta_rp_annotation_mrc_count_list
    )

    # --- Post-Processing: Remove overlaps from bloods_list ---
    # Create a set of all columns in other categories
    all_other_categorized_cols = set()

    # Add all columns from other specific lists to this set
    all_other_categorized_cols.update(annotation_count_list)
    all_other_categorized_cols.update(
        meta_sp_annotation_all_counts
    )  # Use the combined list
    all_other_categorized_cols.update(diagnostic_order_list)
    all_other_categorized_cols.update(drug_order_list)
    all_other_categorized_cols.update(bmi_list)
    all_other_categorized_cols.update(ethnicity_list)
    all_other_categorized_cols.update(annotation_mrc_count_list)
    all_other_categorized_cols.update(
        meta_sp_annotation_mrc_all_counts
    )  # Use the combined list
    all_other_categorized_cols.update(core_02_list)
    all_other_categorized_cols.update(bed_list)
    all_other_categorized_cols.update(vte_status_list)
    all_other_categorized_cols.update(hosp_site_list)
    all_other_categorized_cols.update(core_resus_list)
    all_other_categorized_cols.update(news_list)
    all_other_categorized_cols.update(date_time_stamp_list)
    all_other_categorized_cols.update(appointments_list)

    # Filter bloods_list: keep only elements NOT found in any other category to avoid vte status and others being added to bloods.
    bloods_list = [col for col in bloods_list if col not in all_other_categorized_cols]

    candidate_feature_category_lists = [
        meta_sp_annotation_count_list,
        annotation_count_list,
        diagnostic_order_list,
        drug_order_list,
        bmi_list,
        ethnicity_list,
        annotation_mrc_count_list,
        meta_sp_annotation_mrc_count_list,
        core_02_list,
        bed_list,
        vte_status_list,
        hosp_site_list,
        core_resus_list,
        news_list,
        bloods_list,
        date_time_stamp_list,
        appointments_list,
    ]
    if verbose >= 2:

        data = {}

        for i, lst in enumerate(candidate_feature_category_lists, start=1):
            var_name = [name for name, var in locals().items() if var is lst][0]
            data[var_name] = len(lst)

        plot_candidate_feature_category_lists(data)

    elif verbose >= 1:
        for i, lst in enumerate(candidate_feature_category_lists, start=1):
            var_name = [name for name, var in locals().items() if var is lst][0]
            logger.info("%s: %s", var_name, len(lst))

    pertubation_columns = []

    if local_param_dict.get("data").get("age") == True:
        if "age" in all_df_columns:
            pertubation_columns.append("age")

    if local_param_dict.get("data").get("sex") == True:
        # Find columns containing 'male' instead of hardcoding
        male_cols = [col for col in all_df_columns if "male" in col.lower()]
        pertubation_columns.extend(male_cols)

    if local_param_dict.get("data").get("bmi") == True:
        pertubation_columns.extend(bmi_list)

    if local_param_dict.get("data").get("ethnicity") == True:
        pertubation_columns.extend(ethnicity_list)

    if local_param_dict.get("data").get("bloods") == True:
        pertubation_columns.extend(bloods_list)

    if local_param_dict.get("data").get("diagnostic_order") == True:
        pertubation_columns.extend(diagnostic_order_list)

    if local_param_dict.get("data").get("drug_order") == True:
        pertubation_columns.extend(drug_order_list)

    if local_param_dict.get("data").get("annotation_n") == True:
        pertubation_columns.extend(annotation_count_list)

    if local_param_dict.get("data").get("meta_sp_annotation_n") == True:
        pertubation_columns.extend(meta_sp_annotation_count_list)

    if local_param_dict.get("data").get("annotation_mrc_n") == True:
        pertubation_columns.extend(annotation_mrc_count_list)

    if local_param_dict.get("data").get("meta_sp_annotation_mrc_n") == True:
        pertubation_columns.extend(meta_sp_annotation_mrc_count_list)

    if local_param_dict.get("data").get("core_02") == True:
        pertubation_columns.extend(core_02_list)

    if local_param_dict.get("data").get("bed") == True:
        pertubation_columns.extend(bed_list)

    if local_param_dict.get("data").get("vte_status") == True:
        pertubation_columns.extend(vte_status_list)

    if local_param_dict.get("data").get("hosp_site") == True:
        pertubation_columns.extend(hosp_site_list)

    if local_param_dict.get("data").get("core_resus") == True:
        pertubation_columns.extend(core_resus_list)

    if local_param_dict.get("data").get("news") == True:
        pertubation_columns.extend(news_list)

    if local_param_dict.get("data").get("date_time_stamp") == True:
        pertubation_columns.extend(date_time_stamp_list)

    if local_param_dict.get("data").get("appointments") == True:
        pertubation_columns.extend(appointments_list)

    logger.info(
        "local_param_dict data perturbation: \n %s", local_param_dict.get("data")
    )

    if verbose >= 2:
        plot_dict_values(local_param_dict.get("data"))

    def deduplicate_list(input_list: List[str]) -> List[str]:
        """De-duplicates a list while preserving the original order of elements.

        Args:
            input_list: The list to be de-duplicated.

        Returns:
            A new list with duplicate elements removed, preserving the original order.
        """
        return list(dict.fromkeys(input_list))

    pertubation_columns = deduplicate_list(pertubation_columns)

    # Fallback for generic datasets without specific suffixes
    # Do not trigger fallback if feature_set_n is 99, as this is used for testing the safety net.
    # The disable_fallback flag is used in tests to prevent this fallback from activating.
    # The fallback should trigger if no columns were selected by the toggles.
    all_data_toggles_false = all(
        not v for v in local_param_dict.get("data", {}).values()
    )

    if not pertubation_columns or all_data_toggles_false:
        logger.warning(
            "No columns selected with suffix-based logic. Falling back to generic column selection."
        )
        # Get all columns that are not in the initial drop_list
        potential_columns = [col for col in all_df_columns if col not in set(drop_list)]
        # Also exclude the outcome variable from the feature list
        pertubation_columns = [
            col for col in potential_columns if col != outcome_variable
        ]
        logger.info(
            f"Selected {len(pertubation_columns)} columns using generic fallback logic."
        )

    logger.info("Perturbation columns: %s", pertubation_columns)
    return pertubation_columns, drop_list
