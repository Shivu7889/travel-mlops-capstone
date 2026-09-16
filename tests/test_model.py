from src.classification.train import _load_training_data


def test_training_data_has_single_gender_target():
    train_df = _load_training_data()

    assert "gender" in train_df.columns
    assert not any(column.startswith("gender_") for column in train_df.columns)
