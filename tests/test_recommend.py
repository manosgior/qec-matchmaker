import pandas as pd
import pytest

from qec_matchmaker.recommend import recommend


def predictions(rows, p_above=0.1):
    """One circuit's predictions from (decoder, predicted LER, ms per shot, applicable)"""
    best = 1e-3
    return pd.DataFrame([{"circuit_key": "c", "decoder": d, "p_above": p_above, "ler_q10": best / 3, "ler_q50": best,
                          "ler_q90": best * 3, "regret_q50": ler / best, "ler_decoder_q50": ler, "ms_q10": ms / 2,
                          "ms_q50": ms, "ms_q90": ms * 2, "applicable": ok} for d, ler, ms, ok in rows])


THREE = predictions([("tesseract", 1.0e-3, 100.0, True), ("bposd", 1.2e-3, 10.0, True), ("mwpm", 2.0e-3, 1.0, True)])


def test_accurate_picks_the_lowest_predicted_ler():
    r = recommend(THREE, "accurate")
    assert r.decoder == "tesseract" and r.objective == "accurate" and not r.warnings
    assert r.table.decoder.iloc[0] == "tesseract" and len(r.table) == 3


@pytest.mark.parametrize("tolerance, decoder", [(1.25, "bposd"), (1.1, "tesseract"), (3.0, "mwpm")])
def test_cheapest_within_tolerance(tolerance, decoder):
    r = recommend(THREE, "cheapest", tolerance=tolerance)
    assert r.decoder == decoder and r.table.decoder.iloc[0] == decoder


def test_cheapest_is_the_default():
    assert recommend(THREE).decoder == "bposd"


def test_budget():
    assert recommend(THREE, "budget", budget_ms=50).decoder == "bposd"
    assert recommend(THREE, "budget", budget_ms=500).decoder == "tesseract"
    # Per round: 100 ms per shot over 10 rounds fits 10 ms per round
    assert recommend(THREE, "budget", budget_ms=10, rounds=10).decoder == "tesseract"
    assert recommend(THREE, "budget_per_round", budget_ms=1, rounds=10).decoder == "bposd"


def test_budget_nothing_fits_gives_the_fastest_with_a_warning():
    r = recommend(THREE, "budget", budget_ms=0.5)
    assert r.decoder == "mwpm" and any("budget" in w or "fit" in w for w in r.warnings)


def test_above_threshold_gives_the_cheapest_with_a_warning():
    r = recommend(predictions([("tesseract", 0.5, 100.0, True), ("bposd", 0.5, 10.0, True),
                               ("mwpm", 0.5, 1.0, False)], p_above=0.9), "accurate")
    assert r.decoder == "bposd"
    assert any("above threshold" in w for w in r.warnings)


def test_inapplicable_decoders_are_excluded():
    rows = [("chromobius", 0.5e-3, 0.5, False), ("tesseract", 1.0e-3, 100.0, True), ("bposd", 1.2e-3, 10.0, True),
            ("mwpm", 2.0e-3, 1.0, False)]
    assert recommend(predictions(rows), "accurate").decoder == "tesseract"
    r = recommend(predictions(rows), "cheapest", tolerance=3.0)
    assert r.decoder == "bposd" and set(r.table.decoder) == {"tesseract", "bposd"}
    assert recommend(predictions(rows), "budget", budget_ms=0.1).decoder == "bposd"


def test_decoders_without_predictions_are_left_out_with_a_warning():
    p = THREE.copy()
    p.loc[p.decoder == "mwpm", ["ms_q50", "regret_q50", "ler_decoder_q50"]] = float("nan")
    r = recommend(p, "cheapest", tolerance=3.0)
    assert r.decoder == "bposd" and any("mwpm" in w for w in r.warnings)


def test_no_candidate():
    r = recommend(predictions([("mwpm", 1e-3, 1.0, False)]))
    assert r.decoder is None and r.table.empty


def test_bad_arguments():
    with pytest.raises(ValueError):
        recommend(THREE, "fastest")
    with pytest.raises(ValueError):
        recommend(THREE, "budget")
    with pytest.raises(ValueError):
        recommend(THREE, "budget_per_round", budget_ms=1)
    with pytest.raises(ValueError):
        recommend(pd.concat([THREE, THREE.assign(circuit_key="d")]))
