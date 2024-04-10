import pandas as pd


class read:

    def __init__(self, input_filename):

        filename = input_filename

        print(f"Init main >read on {filename}")

        self.raw_input_data = pd.read_csv(filename)

import pandas as pd

class read_sample:
    def __init__(self, input_filename: str, test_sample_n: int, column_sample_n: int) -> None:
        """
        Initialize the class with the input filename, test sample number, and column sample number.

        :param input_filename: str, the filename of the input data
        :param test_sample_n: int, the number of rows to read from the input data
        :param column_sample_n: int, the number of columns to read from the input data
        :return: None
        """
        self.filename = input_filename

        necessary_columns = ['outcome_var_1', 'age', 'male']

        print(f"Init main > read_sample on {self.filename}")

        if test_sample_n == 0 and column_sample_n == 0:
            self.raw_input_data = None  # Set raw_input_data to None if both parameters are 0
        elif test_sample_n == 0:
            self.raw_input_data = pd.read_csv(self.filename, usecols=lambda x: x in necessary_columns)  # Read only specified columns
        elif column_sample_n == 0:
            self.raw_input_data = pd.read_csv(self.filename, nrows=test_sample_n)  # Read only specified rows
        else:
            self.raw_input_data = pd.read_csv(self.filename, nrows=test_sample_n, usecols=lambda x: x in necessary_columns)  # Read both rows and columns

