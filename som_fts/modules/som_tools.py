import sys
sys.path.append('/home/matheus_cascalho/Documentos/matheus_cascalho/Appliance-Energy-Prediction')
import pandas as pd
from typing import List, NoReturn, Tuple, Callable
from pyFTS.common.Util import persist_obj, sliding_window
from pyFTS.common.transformations.som import SOMTransformation
from datetime import datetime
from som_fts.modules.utils import project_path
from som_fts.modules.artifacts import SOMConfigurations
from pyFTS.models.multivariate.variable import Variable
from pyFTS.partitioners.Grid import GridPartitioner
from pyFTS.models.multivariate.wmvfts import WeightedMVFTS
from pyFTS.benchmarks import Measures
import sys
import traceback
from collections import defaultdict
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler




def reduce_data_by_som(
        df: pd.DataFrame,
        ignore: List[str],
        som_transformator: SOMTransformation,
        endogen_variable: str,
        reduction_file: str
) -> pd.DataFrame:
    cols = [col for col in df.columns if col not in ignore]
    to_reduce = df[cols]
    scaler = MinMaxScaler()
    to_reduce = pd.DataFrame(scaler.fit_transform(to_reduce), columns=cols)
    reduced = som_transformator.apply(data=to_reduce, endogen_variable=endogen_variable)
    reduced.to_csv(reduction_file, index=False)
    return reduced


def create_som_transformator(
        data: pd.DataFrame,
        grid_dimension: int,
        endogen_variable: str,
        ignore: List[str],
        epochs: int,
        learning_rate: float,
        train_percentage: float = 0.75,
        filename: str = 'data/transformators/som_transformator.som'
) -> SOMTransformation:
    gd = (grid_dimension, grid_dimension)
    som = SOMTransformation(grid_dimension=gd)
    cols = [col for col in data.columns if col != endogen_variable and col not in ignore]
    df = data[cols]
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df[cols].to_numpy()), columns=cols)
    som.train(data=df, epochs=epochs, percentage_train=train_percentage, leaning_rate=learning_rate)
    persist_obj(som, filename)
    print(f"\nSom transformator saved as {filename}\n")
    return som


def train_som_and_reduce_dataset_varying_grids(
        grid: int,
        som_config: SOMConfigurations,
        interval: Tuple[int, int],
        learning_rate: float,
        prefix: str = "",
        sufix: str = ""
) -> NoReturn:
    head_message = "=" * 100
    head_message += f"Params:\n\tGrid: {(grid, grid)}\n\tEpochs: {som_config.epochs}\n\tpartitions: {partitions}\n\tinterval: {interval}"
    head_message += f"\n\tsufix: {sufix}"
    start_time = datetime.now()
    head_message += f"\n{start_time} -- START"

    print(head_message)

    transformator_file = project_path(f'data/som_transformators/cross_validation/{prefix}transformator_{grid}_{som_config.epochs}_epochs.som')

    transformator = create_som_transformator(
        data=som_config.data,
        grid_dimension=grid,
        endogen_variable=som_config.endogen_variable,
        ignore=som_config.ignore,
        epochs=som_config.epochs,
        train_percentage=0.75,
        filename=transformator_file,
        learning_rate=learning_rate
    )
    training_time = datetime.now()
    head_message += f"\n{training_time} -- finish training -- {training_time - start_time} to train"

    reduction_file = project_path(f'data/som_reductions_validations/{prefix}reduction_{grid}_{som_config.epochs}_epochs{sufix}.csv')
    reduced_data = reduce_data_by_som(
        df=som_config.data,
        ignore=som_config.ignore,
        som_transformator=transformator,
        endogen_variable=som_config.endogen_variable,
        reduction_file=reduction_file
    )

    return reduced_data


def create_and_train_models_with_train_test_split(
        data: pd.DataFrame,
        som_config: SOMConfigurations,
        spliter: Callable,
        partitions: Tuple[int, int, int] = (50, 50, 50),
        models_folder: str = project_path('data/fts_models'),
        prefix: str = "",
) -> NoReturn:

    head_message = "=" * 100
    start_time = datetime.now()
    head_message += f"\n{start_time} -- START"

    print(head_message)

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
    filename = f"{prefix}_model_train_with_{train_percentage}_percent_{partitions[0]}_partitions.model"
    filepath = f"{models_folder}/{filename}"

    persist_obj(model, filepath)

    rmse, mape, u = Measures.get_point_statistics(
        data=test,
        model=model
    )
    forecast = model.predict(test)
    smape = Measures.smape(test[som_config.endogen_variable], forecast)
    mae = mean_absolute_error(test[som_config.endogen_variable], forecast)

    finish = datetime.now()
    finish_message = f"\nFinish Training -- {finish - start_time} to train FTS model"
    finish_message += f"\n\tRMSE: {rmse}\n\tMAPE: {mape}\n\tU: {u}"

    print(finish_message)

    data = {
        partitions: {
                "rmse": rmse,
                "mape": mape,
                "u": u,
                "smape": smape,
                "mae": mae,
                "time to train FTS model": finish - start_time,
                "rules": len(model),
                "train percentage": train_percentage,
                "FTS model file": filepath
            }
    }
    return data


if __name__=="__main__":
    # from sklearn.preprocessing import MinMaxScaler

    # filename = project_path('data/HomeC/sanitized_homeCsub_amostrado.csv')
    filename = project_path('data/energydata_complete.csv')

    # sub_sampled = filename[:-4] + "sub_amostrado.csv"
    data = pd.read_csv(filename)



    # for col in data.columns[2:]:
    #     data[col] = data[col].apply(lambda x: float('nan') if  x == "?" else float(x))
    # data.dropna(inplace=True)
    # data = data.iloc[lambda x: x.index%10 == 0]
    # data.to_csv(sub_sampled, index=False)

    #
    # data =
    # data.pop('date')

    n_windows = 30
    tests_by_step = 5
    window = round(len(data) / n_windows)
    last_step = 0
    grids = [20]#, 100, 35, 20]
    partitions = (40, 30, 20, 50, 10)
    reduction_results = []
    forecast_results = defaultdict(list)
    forecast_result = defaultdict(list)

    failures = []
    epochs = 50

    for epochs in [70, 80, 90, 100]:
        for learning_rate in [0.1, 0.0001, 0.00001]:
            for grid in grids:
                data = pd.read_csv(filename)
                for col in data.columns[2:]:
                    data[col] = data[col].apply(lambda x: float('nan') if x == "?" else float(x))
                data.dropna(inplace=True)
                for ct, (_, train, test) in enumerate(sliding_window(data, window, 0.75, inc=1)):
                    li = ct * window
                    ls = (ct+1) * window
                    DATASET = f"APPLIANCE_ep{epochs}_rate{learning_rate}"
                    pd.DataFrame(reduction_results).to_csv(f"{DATASET}_reduction_results_grid{grid}.csv", index=False)
                    for partitions, fr in forecast_results.items():
                        pd.DataFrame(fr).to_csv(f'{DATASET}_forecast_results_grid{grid}_{partitions[0]}partitions.csv', index=False)

                    home_c_ignore = ['time', 'icon', 'summary', 'cloudCover']
                    home_c_endogen = 'use [kW]'

                    householder_ignore = ['Date', 'Time']
                    householder_endogen = 'Global_active_power'

                    appliance_ignore = ['date']
                    appliance_endogen = 'Appliances'

                    som_config = SOMConfigurations(
                        data=data.iloc[li:ls],
                        endogen_variable=appliance_endogen,
                        epochs=epochs,
                        ignore=appliance_ignore
                    )

                    reduced_data = train_som_and_reduce_dataset_varying_grids(
                        grid=grid,
                        som_config=som_config,
                        prefix=f"Household_",
                        interval=(li, ls),
                        sufix=f"_window_{ct}",
                        learning_rate=learning_rate
                    )

                    print("=-=" * 50)
                    forecast_result = {}
                    for partition in [10,20,30,40,50]:
                        forecast_result.update(
                            create_and_train_models_with_train_test_split(
                                data=reduced_data,
                                som_config=som_config,
                                spliter=lambda x: (x[:round(len(x)*0.75)], x[round(len(x)*0.75):]),
                                partitions=(partition, partition, partition),
                                prefix=f"Household_window_{ct}_",
                            )
                        )
                    for fr in forecast_result:
                        forecast_result[fr].update(
                            {
                                "partitions": fr,
                                "interval": (li, ls),
                                "grid": grid,
                                "id": f"Household_report_{grid}_{som_config.epochs}_epochs_window_{ct}_{fr}_partitions"
                            }
                        )
                        forecast_results[fr].append(forecast_result[fr])

