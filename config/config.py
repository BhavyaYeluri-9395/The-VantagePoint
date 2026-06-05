DATA_PATH = "data/movie_dataset_MEGA_3000plus.csv"

FEATURE_COLUMNS = [
    "budget_usd",
    "revenue_usd",
    "popularity_score",
    "avg_movie_rating",
    "cast_popularity_score",
    "fuzzy_score"
]

TARGET_COLUMN = "avg_user_rating"

TEST_SIZE = 0.2
RANDOM_STATE = 42