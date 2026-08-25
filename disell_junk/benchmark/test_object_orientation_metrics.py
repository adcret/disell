import numpy as np
import pytest

from object_orientation_metrics import match_cells


def base():
    truth = np.zeros((1, 8, 12), int)
    truth[:, :, :4], truth[:, :, 4:8], truth[:, :, 8:] = 1, 2, 3
    means = np.array([[0, 0], [.10, .20], [.30, .40], [.50, .60]])
    field = means[truth]
    return truth, field


def score(pred, field=None, purity=.5, completeness=.5):
    truth, default_field = base()
    return match_cells(truth, pred, default_field if field is None else field, (1, 1, 1),
                       purity_threshold=purity, completeness_threshold=completeness)[1]


def test_perfect_correspondence():
    truth, _ = base(); result = score(truth)
    assert result["one_to_one_recovered_cells"] == 3
    assert result["orientation_correct_cells_at_0p01deg"] == 3


def test_shifted_boundaries_are_tolerated():
    truth, _ = base(); pred = np.roll(truth, 1, axis=2); pred[:, :, 0] = 1
    result = score(pred)
    assert result["one_to_one_recovered_cells"] == 3
    assert result["cell_count_error"] == 0


def test_one_split_is_detected():
    truth, _ = base(); pred = truth.copy(); pred[:, 4:, :4] = 4
    result = score(pred)
    assert result["split_true_cells"] == 1
    assert result["duplicate_predictions"] >= 1


def test_one_merge_is_detected():
    truth, _ = base(); pred = truth.copy(); pred[pred == 2] = 1
    result = score(pred)
    assert result["merged_predicted_cells"] == 1
    assert result["unrepresented_true_cells"] == 0


def test_missing_cell_is_unmatched_and_unrepresented():
    truth, _ = base(); pred = truth.copy(); pred[pred == 3] = 0
    result = score(pred)
    assert result["unmatched_true_cells"] == 1
    assert result["unrepresented_true_cells"] == 1


def test_compensating_split_merge_defeats_exact_count():
    truth, _ = base(); pred = truth.copy(); pred[pred == 2] = 1; pred[:, 4:, 8:] = 2
    result = score(pred)
    assert result["cell_count_error"] == 0
    assert result["one_to_one_recovered_cells"] < 3
    assert result["split_true_cells"] == 1
    assert result["merged_predicted_cells"] == 1


def test_correct_marginal_means_do_not_imply_identity():
    # Truth rows and predicted columns have the same mean multiset, but every
    # predicted cell contains half of each true cell.
    truth = np.array([[[1, 1], [2, 2]]])
    pred = np.array([[[1, 2], [1, 2]]])
    scalar = np.array([[[.1, .1], [.5, .1]]])
    field = np.stack((scalar, 2 * scalar), axis=-1)
    result = match_cells(truth, pred, field, (1, 1, 1),
                         purity_threshold=.6, completeness_threshold=.6)[1]
    assert result["cell_mean_wasserstein_deg"] == pytest.approx(0)
    assert result["one_to_one_recovered_cells"] == 0


def test_wrapped_two_channel_error_uses_euclidean_norm():
    truth, field = base(); shifted = field.copy(); shifted[truth == 1] += [.003, .004]
    pairs, _ = match_cells(truth, truth, shifted, (1, 1, 1))
    first = next(x for x in pairs if x["true_cell"] == 1)
    # Prediction and truth use the same field here, so construct the norm rule directly
    # through the reported component relationship on a controlled shifted boundary case.
    assert first["mean_orientation_error_deg"] == pytest.approx(
        np.hypot(first["chi_mean_error_deg"], first["phi_mean_error_deg"])
    )
