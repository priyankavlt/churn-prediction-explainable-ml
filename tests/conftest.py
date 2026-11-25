import json
from pathlib import Path

import pandas as pd
import pytest

from churn.config import DATA_PATH, ID_COL, MODEL_PATH, TARGET


@pytest.fixture(scope="session")
def raw_df():
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="session")
def customers(raw_df):
    return raw_df.drop(columns=[ID_COL, TARGET])


@pytest.fixture
def customer():
    return json.loads((Path(__file__).parent / "sample_customer.json").read_text())


requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="trained model not found; run `make train`"
)
