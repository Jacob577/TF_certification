import numpy as np
import pandas as pd 
import MetaTrader5 as mt5
from datetime import datetime
# from sklearn.preprocessing import MinMaxScaler

class Data:
    def __init__(self, symbol: str, from_date: datetime, to_date: datetime):
    # def __init__(self, symbol="EURUSD", from_date=datetime(2020, 1, 1), 
    #              to_date=datetime(2023, 5, 31)):
        self.dataset = None
        self.symbol = symbol
        self.from_date = from_date
        self.to_date = to_date
        self.column_list = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
        

    def set_to_from(self, to_date, from_date):
       self.to_date = to_date
       self.from_date = from_date

    def fetch_data(self):
        if not mt5.initialize():
            print("initialize() failed, error code =",mt5.last_error())
            mt5.quit()

        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_H1, self.from_date, self.to_date)
        df = pd.DataFrame(rates, columns=self.column_list)

        self.dataset = df
    
    def normalize_data(self, depth: int, foreseeing: int, column_titel: str):
        self.fetch_data()
        list_normalized = []
        valid_list = []
        for i in range(len(self.dataset), 0, -1):
            if i > depth and len(self.dataset) - i > foreseeing:                
                valid_list = [a for a in self.dataset[column_titel][i:i+foreseeing] ]
                list_normalized.append(self.normalize_list(self.dataset[column_titel][i-depth:i], valid_list))

        return list_normalized

    def normalize_list(self, list_values, valid):
        max = list_values.max()
        min = list_values.min()
        new_list_normalized = []
        valid_list_normalized =[]
        for i in list_values:
            new_list_normalized.append((i - min)/(max - min))
        
        for j in valid:
            valid_list_normalized.append((j-min)/(max-min))
            
        return [new_list_normalized, valid_list_normalized]

    def get_original_data(self):
       self.fetch_data()
       return self.dataset

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])