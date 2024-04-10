import pandas as pd


class read:

    def __init__(self, input_filename):

        filename = input_filename

        print(f"Init main >read on {filename}")

        self.raw_input_data = pd.read_csv(filename)

class read_sample:
    def __init__(self, input_filename, test_sample_n, column_sample_n):
        self.filename = input_filename
        print(f"Init main > read_sample on {self.filename}")

        if test_sample_n == 0 and column_sample_n == 0:
            self.raw_input_data = None  # Set raw_input_data to None if both parameters are 0
        elif test_sample_n == 0:
            self.raw_input_data = pd.read_csv(self.filename, usecols=range(column_sample_n))  # Read only specified columns
        elif column_sample_n == 0:
            self.raw_input_data = pd.read_csv(self.filename, nrows=test_sample_n)  # Read only specified rows
        else:
            self.raw_input_data = pd.read_csv(self.filename, nrows=test_sample_n, usecols=range(column_sample_n))  # Read both rows and columns
