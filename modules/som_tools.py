import pandas as pd
from typing import List
from pyFTS.common.Util import persist_obj, load_obj
from pyFTS.common.transformations.som import SOMTransformation
from math import ceil
from modules.utils import project_path


def reduce_data_by_som(
        df: pd.DataFrame,
        ignore: List[str],
        som_transformator: SOMTransformation,
        endogen_variable: str,
        reduction_file: str
) -> pd.DataFrame:
    cols = [col for col in df.columns if col not in ignore]
    to_reduce = df[cols]
    reduced = som_transformator.apply(data=to_reduce, endogen_variable=endogen_variable)
    reduced.to_csv(reduction_file)
    return reduced


def create_som_transformator(
        data: pd.DataFrame,
        grid_dimension: int,
        endogen_variable: str,
        ignore: List[str],
        epochs: int,
        train_percentage: float = 0.75,
        filename: str = 'data/transformators/som_transformator.som'
) -> SOMTransformation:
    gd = (grid_dimension, grid_dimension)
    som = SOMTransformation(grid_dimension=gd)
    cols = [col for col in data.columns if col != endogen_variable and col not in ignore]
    df = data[cols]
    som.train(data=df, epochs=epochs, percentage_train=train_percentage)
    persist_obj(som, filename)
    print(f"Som transformator saved as {filename}")
    return som

