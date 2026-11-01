import pandas as pd
from src.benchmark import FEATURES, TARGET, make_preprocessor

def test_dataset_contract():
    assert TARGET == "class"
    assert FEATURES == [
        "Region", "Fresh", "Milk", "Grocery", "Frozen",
        "Detergents_Paper", "Delicassen"
    ]

def test_preprocessing_pipeline_runs():
    sample = pd.DataFrame({
        "Region": [1, 2, 1, 3],
        "Fresh": [10, 20, 30, 40],
        "Milk": [5, 6, 7, 8],
        "Grocery": [2, 3, 4, 5],
        "Frozen": [1, 2, 3, 4],
        "Detergents_Paper": [1, 2, 3, 4],
        "Delicassen": [0, 1, 0, 1],
    })
    result = make_preprocessor(scale_numeric=True).fit_transform(sample)
    assert result.shape[0] == len(sample)
    assert result.shape[1] > len(FEATURES)
