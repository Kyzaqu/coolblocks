import pandas as pd
import pytest

from coolblocks.prioritization.score import minmax, priority_score


def _df():
    return pd.DataFrame({"lst_mean": [30.0, 40.0, 35.0], "svi_overall": [0.9, 0.1, 0.5]})


def test_minmax_range():
    s = minmax(pd.Series([1, 2, 3]))
    assert s.min() == 0 and s.max() == 1


def test_minmax_constant():
    assert minmax(pd.Series([5, 5, 5])).sum() == 0


def test_heat_only_ranks_hottest_first():
    out = priority_score(_df(), {"lst_mean": 1.0})
    assert out.iloc[0]["lst_mean"] == 40.0 and out.iloc[0]["priority_rank"] == 1


def test_weights_change_ranking():
    heat = priority_score(_df(), {"lst_mean": 1.0, "svi_overall": 0.0}).iloc[0]["lst_mean"]
    svi = priority_score(_df(), {"lst_mean": 0.0, "svi_overall": 1.0}).iloc[0]["svi_overall"]
    assert heat == 40.0 and svi == 0.9


def test_weights_are_scale_invariant():
    a = priority_score(_df(), {"lst_mean": 1, "svi_overall": 1})["priority_score"].tolist()
    b = priority_score(_df(), {"lst_mean": 5, "svi_overall": 5})["priority_score"].tolist()
    assert a == pytest.approx(b)


@pytest.mark.parametrize("w", [{}, {"lst_mean": -1}, {"lst_mean": 0}, {"nope": 1}])
def test_bad_weights(w):
    with pytest.raises((ValueError, KeyError)):
        priority_score(_df(), w)
