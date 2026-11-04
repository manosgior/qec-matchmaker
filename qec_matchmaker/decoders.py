"""The decoders Matchmaker knows (named as in ECCentric), and when each one applies to a circuit."""
import math

DECODERS = {
    "mwpm": "minimum-weight perfect matching (PyMatching)",
    "bposd": "BP with ordered-statistics decoding (ldpc)",
    "bplsd": "BP with localized-statistics decoding (ldpc)",
    "belief_find": "BP with union-find (ldpc)",
    "relay_bp": "Relay-BP",
    "tesseract": "Tesseract search decoder",
    "chromobius": "Chromobius color-code decoder",
}


def _flag(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else bool(value)


def applicable(features: dict) -> dict:
    """Which decoders can decode a circuit with these features. Matching needs errors that decompose
    into edges (known from a circuit; from a DEM alone, only graph-like errors are certain); Chromobius
    needs color-annotated detectors"""
    decomposable = _flag(features.get("decomposable"))
    if decomposable is None:
        decomposable = _flag(features.get("graphlike")) or False
    return {name: (decomposable if name == "mwpm" else bool(_flag(features.get("color_annotated")))
                   if name == "chromobius" else True)
            for name in DECODERS}
