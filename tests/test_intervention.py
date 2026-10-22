import numpy as np

from metaderm.reasoning.intervention import apply_oracle_correction


def test_oracle_correction_is_one_hot_and_does_not_mutate_input():
    probs = {
        "pigment_network": np.array([[0.1, 0.7, 0.2]], dtype=float),
        "streaks": np.array([[0.8, 0.1, 0.1]], dtype=float),
        "pigmentation": np.array([[0.5, 0.2, 0.1, 0.1, 0.1]], dtype=float),
        "regression_structures": np.array([[0.7, 0.1, 0.1, 0.1]], dtype=float),
        "dots_and_globules": np.array([[0.6, 0.2, 0.2]], dtype=float),
        "blue_whitish_veil": np.array([[0.9, 0.1]], dtype=float),
        "vascular_structures": np.array([[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02]], dtype=float),
    }

    original = probs["pigment_network"].copy()
    corrected = apply_oracle_correction(
        probs,
        sample_index=0,
        concept="pigment_network",
        true_state=2,
    )

    assert np.allclose(probs["pigment_network"], original)
    assert np.allclose(corrected["pigment_network"], [[0.0, 0.0, 1.0]])
    assert np.isclose(corrected["pigment_network"].sum(), 1.0)
