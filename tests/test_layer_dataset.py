"""Synthetic tests for disell.layer_dataset: numeric sorting, validation,
cropping, masking against the motor range."""

import json

import numpy as np
import pytest

from disell import LayerDatasetError, discover_layers, load_layer_volume


def make_dataset(tmp_path, n_layers=11, shape=(20, 30), C=2, scramble=False):
    """Build a synthetic layer_<n>_1 dataset. Channel c value range c*10+[0,1]."""
    motors = np.stack(
        [np.linspace(0, 1, 5 * 7).reshape(5, 7) + 10 * c for c in range(C)]
    )
    rng = np.random.default_rng(0)
    order = list(range(1, n_layers + 1))
    for n in order:
        d = tmp_path / f"layer_{n}_1"
        d.mkdir()
        arr = np.stack(
            [rng.uniform(10 * c, 10 * c + 1, size=shape) for c in range(C)],
            axis=-1,
        )
        # plant one failed fit (0.0 in channel 1, far outside its motor range)
        arr[1, 2, 1] = 0.0
        np.save(d / "mean.npy", arr)
        np.save(d / "motors.npy", motors)
        (d / "processing_info.json").write_text(json.dumps({"scan_id": f"{n}.1"}))
        np.save(d / "segmap.npy", np.ones(shape))
    return tmp_path


def test_discover_layers_numeric_sort(tmp_path):
    make_dataset(tmp_path, n_layers=11)
    layers = discover_layers(tmp_path)
    nums = [n for n, _ in layers]
    assert nums == list(range(1, 12))  # 10 and 11 after 2..9, not lexicographic
    names = [p.name for _, p in layers]
    assert names[0] == "layer_1_1" and names[-1] == "layer_11_1"


def test_load_layer_volume_stacks_and_masks(tmp_path):
    make_dataset(tmp_path, n_layers=3, shape=(20, 30))
    vol = load_layer_volume(tmp_path, spacing_nm=(500, 1240, 400), expect_n_layers=3)
    assert vol.field.shape == (3, 20, 30, 2)
    assert vol.field.dtype == np.float32
    assert vol.mask.shape == (3, 20, 30)
    # the planted failed fit is masked, everything else valid
    assert not vol.mask[:, 1, 2].any()
    assert vol.mask.sum() == 3 * 20 * 30 - 3
    assert vol.spacing_nm == (500.0, 1240.0, 400.0)
    assert vol.channel_names == ("chi", "phi")
    assert vol.layer_names[0] == "layer_1_1"


def test_load_layer_volume_crop(tmp_path):
    make_dataset(tmp_path, n_layers=2, shape=(20, 30))
    vol = load_layer_volume(
        tmp_path,
        spacing_nm=(1, 1, 1),
        crop=(5, 15, 10, 30),
        expect_shape_yx=(10, 20),
    )
    assert vol.field.shape == (2, 10, 20, 2)
    assert vol.source["crop"] == [5, 15, 10, 30]


def test_load_layer_volume_shape_mismatch_raises(tmp_path):
    make_dataset(tmp_path, n_layers=3)
    bad = np.zeros((21, 30, 2))
    np.save(tmp_path / "layer_2_1" / "mean.npy", bad)
    with pytest.raises(LayerDatasetError, match="differ in shape"):
        load_layer_volume(tmp_path, spacing_nm=(1, 1, 1))


def test_load_layer_volume_motor_mismatch_raises(tmp_path):
    make_dataset(tmp_path, n_layers=3)
    motors = np.load(tmp_path / "layer_3_1" / "motors.npy")
    np.save(tmp_path / "layer_3_1" / "motors.npy", motors + 0.5)
    with pytest.raises(LayerDatasetError, match="motors"):
        load_layer_volume(tmp_path, spacing_nm=(1, 1, 1))


def test_load_layer_volume_missing_file_raises(tmp_path):
    make_dataset(tmp_path, n_layers=2)
    (tmp_path / "layer_1_1" / "processing_info.json").unlink()
    with pytest.raises(LayerDatasetError, match="missing file"):
        load_layer_volume(tmp_path, spacing_nm=(1, 1, 1))


def test_load_layer_volume_wrong_layer_count_raises(tmp_path):
    make_dataset(tmp_path, n_layers=2)
    with pytest.raises(LayerDatasetError, match="expected 11 layers"):
        load_layer_volume(tmp_path, spacing_nm=(1, 1, 1), expect_n_layers=11)


def test_segmaps_loaded_but_not_used_as_mask(tmp_path):
    make_dataset(tmp_path, n_layers=2)
    vol = load_layer_volume(tmp_path, spacing_nm=(1, 1, 1), load_segmaps=True)
    assert vol.segmaps.shape == (2, 20, 30)
    # mask is unaffected by segmap content
    assert vol.mask.sum() == 2 * 20 * 30 - 2
