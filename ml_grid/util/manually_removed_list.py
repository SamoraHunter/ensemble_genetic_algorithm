from typing import List


class ListHolder:
    """
    A container for a manually curated list of features to be excluded from analysis.

    This class serves as a centralized, hardcoded repository for column names
    that have been identified for removal for specific reasons not covered by
    automated cleaning steps (e.g., known data entry errors, irrelevant
    identifiers, or features deemed inappropriate for the model).
    """

    feature_list: List[str]
    """A list of feature names to be manually removed."""

    def __init__(self) -> None:
        """Initializes the ListHolder and populates the feature list."""

        self.feature_list = []
