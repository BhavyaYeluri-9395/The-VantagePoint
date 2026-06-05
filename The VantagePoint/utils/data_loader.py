import pandas as pd
from sklearn.model_selection import train_test_split
from config.config import DATA_PATH, TEST_SIZE, RANDOM_STATE

def load_data():

    df = pd.read_csv(DATA_PATH)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    return train_df, test_df