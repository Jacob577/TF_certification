import MetaTrader5 as mt5
from datetime import datetime
from data import Data

if __name__ == "__main__":
    data = Data(from_date=datetime(2022, 1, 1), to_date=datetime(2022, 1, 15))
    print(data.normalize_data(depth=40, foreseeing=10, column_titel="close"))
