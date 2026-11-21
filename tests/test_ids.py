import pandas as pd
import pytest

from coolblocks.features.ids import add_tract_id, tract_id_from_unit_id


def test_tract_id_is_first_11_digits():
    s = pd.Series(["060375701001", "060375701002", "060375702003"])
    assert tract_id_from_unit_id(s).tolist() == ["06037570100", "06037570100", "06037570200"]


def test_leading_zero_preserved():
    assert tract_id_from_unit_id(pd.Series(["060375701001"])).iloc[0].startswith("06")


def test_rejects_wrong_length():
    with pytest.raises(ValueError):
        tract_id_from_unit_id(pd.Series(["6037570100"]))  # leading zero was lost


def test_add_tract_id_keeps_other_columns():
    df = add_tract_id(pd.DataFrame({"unit_id": ["060375701001"], "x": [1]}))
    assert list(df.columns) == ["unit_id", "x", "tract_id"]
