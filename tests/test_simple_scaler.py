import numpy as np
import pytest

from modeling import SimpleStandardScaler


@pytest.mark.parametrize(
    "X",
    [
        np.array([[1.0], [2.0], [3.0]]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[10.0], [20.0], [30.0]]),
    ],
)
def test_simple_standard_scaler(X):
    scaler = SimpleStandardScaler()
    X_scaled = scaler.fit_transform(X)

    # After standardization: mean ≈ 0, std ≈ 1
    assert np.allclose(X_scaled.mean(axis=0), 0.0)
    assert np.allclose(X_scaled.std(axis=0), 1.0)
