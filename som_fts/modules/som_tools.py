import pandas as pd
from typing import List, NoReturn, Tuple, Callable
from pyFTS.common.Util import persist_obj
from pyFTS.common.transformations.som import SOMTransformation
from datetime import datetime
from som_fts.modules.utils import project_path
from som_fts.modules.artifacts import SOMConfigurations
from pyFTS.models.multivariate.variable import Variable
from pyFTS.partitioners.Grid import GridPartitioner
from pyFTS.models.multivariate.wmvfts import WeightedMVFTS
from pyFTS.benchmarks import Measures


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
    reduced.to_csv(reduction_file, index=False)
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
        partitions: Tuple[int, int, int] = (50, 50, 50),
        prefix: str = ""
) -> NoReturn:
    for gd in grids:
        head_message = "=" * 100
        head_message += f"Params:\n\tGrid: {(gd, gd)}\n\tEpochs: {som_config.epochs}\n\tpartitions: {partitions}"
        start_time = datetime.now()
        head_message += f"\n{start_time} -- START"

        print(head_message)

        transformator_file = project_path(f'data/som_transformators/{prefix}transformator_{gd}_{som_config.epochs}_epochs.som')

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

        reduction_file = project_path(f'data/som_reductions/{prefix}reduction_{gd}_{som_config.epochs}_epochs.csv')
        reduced_df = reduce_data_by_som(
            df=som_config.data,
            ignore=som_config.ignore,
            som_transformator=transformator,
            endogen_variable=som_config.endogen_variable,
            reduction_file=reduction_file
        )

        projection_time = datetime.now()
        head_message += f"\n{projection_time} -- finish projection -- {projection_time - training_time} to project"

        report_file = project_path(f'data/reports/{prefix}report_{gd}_{som_config.epochs}_epochs.txt')
        with open(report_file, 'w') as rep:
            rep.write(head_message)
        head_message = ''
        # print(reduced_df.head())


def create_and_train_models_with_train_test_split(
        grids: List[int],
        som_config: SOMConfigurations,
        spliter: Callable,
        partitions: Tuple[int, int, int] = (50, 50, 50),
        reductions_folder: str = project_path('data/som_reductions'),
        models_folder: str = project_path('data/fts_models'),
        prefix: str = ""
) -> NoReturn:
    for gd in grids:
        head_message = "=" * 100
        head_message += f"Params:\n\tGrid: {(gd, gd)}\n\tEpochs: {som_config.epochs}\n\tpartitions: {partitions}\n"
        start_time = datetime.now()
        head_message += f"\n{start_time} -- START"

        print(head_message)

        filename = f"{prefix}reduction_{gd}_{som_config.epochs}_epochs.csv"
        data = pd.read_csv(f"{reductions_folder}/{filename}")
        data[som_config.endogen_variable] = som_config.data[som_config.endogen_variable]

        x = Variable(
            "x",
            data_label="x",
            partitioner=GridPartitioner,
            npart=partitions[0],
            data=data
        )

        y = Variable(
            "y",
            data_label="y",
            partitioner=GridPartitioner,
            npart=partitions[1],
            data=data
        )

        z = Variable(
            name=som_config.endogen_variable,
            data_label=som_config.endogen_variable,
            partitioner=GridPartitioner,
            npart=partitions[2],
            data=data
        )

        model = WeightedMVFTS(
            explanatory_variables=[x, y, z],
            target_variable=z
        )

        # train_limit = ceil(len(data) * train_percentage)

        train, test = spliter(data)

        print(f"Train length: {len(train)}")
        print(f"Test length: {len(test)}\n")

        model.fit(ndata=train, dump="time")
        train_percentage = round(100 * len(train) / len(data))
        filename = f"{prefix}_model_{gd}_train_with_{train_percentage}_percent_{partitions[0]}_partitions.model"
        filepath = f"{models_folder}/{filename}"

        persist_obj(model, filepath)

        rmse, mape, u = Measures.get_point_statistics(
            data=test,
            model=model
        )

        finish = datetime.now()
        finish_message = f"\nFinish Training -- {finish - start_time} to train FTS model"
        finish_message += f"\n\tRMSE: {rmse}\n\tMAPE: {mape}\n\tU: {u}"

        print(finish_message)

        report_name = project_path(
            f"data/reports/{prefix}_report_FTS_{gd}_SOM_with_{train_percentage}_percent_percent_{partitions[0]}_partitions.txt"
        )
        with open(report_name, 'w') as report:
            report.write(head_message + finish_message)


if __name__=="__main__":
    from sklearn.preprocessing import MinMaxScaler

    filename = project_path('data/energydata_complete.csv')
    data = pd.read_csv(filename)
    data.pop('date')
    scaled = MinMaxScaler()
    scaled_data = scaled.fit_transform(data)
    scaled_data = pd.DataFrame(columns=data.columns, data=scaled_data)
    scaled_data['Appliances'] = data['Appliances']

    som_config = SOMConfigurations(
        data=data,
        endogen_variable='Appliances',
        epochs=10000,
        ignore=['date']
    )
    # pipeline(
    #     grids=[25, 35, 50, 100],
    #     som_config=som_config,
    #     partitions=(50, 50, 50),
    #     prefix="SCALED_"
    # )

    print("=-="*50)

    create_and_train_models_with_train_test_split(
        grids=[25, 35, 50, 100],
        som_config=som_config,
        spliter=lambda x: (x[:round(len(x)*0.75)], x[round(len(x)*0.75):]),
        partitions=(25, 25, 25),
        prefix="SCALED_"
    )
