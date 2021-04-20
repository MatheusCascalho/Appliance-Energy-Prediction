import pandas as pd
from typing import List, NoReturn, Tuple
from pyFTS.common.Util import persist_obj, load_obj
from pyFTS.common.transformations.som import SOMTransformation
import time
from datetime import datetime
from dataclasses import dataclass
from math import ceil
from modules.utils import project_path
from modules.artifacts import SOMConfigurations


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
    print(f"\nSom transformator saved as {filename}\n")
    return som


def pipeline(
        grids: List[int],
        som_config: SOMConfigurations,
        partitions: Tuple[int, int, int] = (50, 50, 50)
) -> NoReturn:
    for gd in grids:
        head_message = "=" * 100
        head_message += f"Params:\n\tGrid: {(gd, gd)}\n\tEpochs: {som_config.epochs}\n\tpartitions: {partitions}"
        start_time = datetime.now()
        head_message += f"\n{start_time} -- START"

        print(head_message)

        transformator_file = project_path(f'data/som_transformators/transformator_{gd}_{som_config.epochs}_epochs.som')

        transformator = create_som_transformator(
            data=som_config.data,
            grid_dimension=gd,
            endogen_variable=som_config.endogen_variable,
            ignore=som_config.ignore,
            epochs=som_config.epochs,
            train_percentage=0.75,
            filename=transformator_file
        )
        training_time = datetime.now()
        head_message += f"\n{training_time} -- finish training -- {training_time - start_time} to train"

        reduction_file = project_path(f'data/som_reductions/reduction_{gd}_{som_config.epochs}_epochs.csv')
        reduced_df = reduce_data_by_som(
            df=som_config.data,
            ignore=som_config.ignore,
            som_transformator=transformator,
            endogen_variable=som_config.endogen_variable,
            reduction_file=reduction_file
        )

        projection_time = datetime.now()
        head_message += f"\n{projection_time} -- finish projection -- {projection_time - training_time} to project"

        report_file = project_path(f'data/reports/report_{gd}_{som_config.epochs}_epochs.txt')
        with open(report_file, 'w') as rep:
            rep.write(head_message)
        head_message = ''
        # print(reduced_df.head())


if __name__=="__main__":
    filename = project_path('data/energydata_complete.csv')
    data = pd.read_csv(filename)
    som_config = SOMConfigurations(
        data=data,
        endogen_variable='Appliances',
        epochs=1,
        ignore=['date']
    )
    pipeline(
        grids=[2, 3],
        som_config=som_config,
        partitions=(2, 2, 2)
    )