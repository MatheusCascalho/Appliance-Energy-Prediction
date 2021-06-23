import pandas as pd
from pca_fts.PcaWeightedMVFTS import PcaWeightedMVFTS
from pyFTS.benchmarks import Measures
import matplotlib.pyplot as plt

def sample_first_prows(data, perc=0.75):
    return data.head(int(len(data)*(perc)))

filename = '/home/hugo/projetos-doutorado/Appliance-Energy-Prediction/data/energydata_complete.csv'
data = pd.read_csv(filename)
data.pop('date')
data.pop('rv1')
data.pop('rv2')

train = sample_first_prows(data,0.75)
test = data.iloc[max(train.index):]
y_test = data.iloc[max(train.index):]['Appliances'].values

pca_wmvfts = PcaWeightedMVFTS(n_components = 2,
                       endogen_variable = 'Appliances',
                       n_part = 20)

model, pca_reduced_train = pca_wmvfts.run_train_model(train)
forecast, forecast_self, pca_reduced_test = pca_wmvfts.run_test_model(model,test)


print("RMSE")
print(Measures.rmse(pca_reduced_test['Appliances'],forecast))

print("MAPE")
print(Measures.mape(pca_reduced_test['Appliances'],forecast))

rmse, mape, u = Measures.get_point_statistics(
    data=pca_reduced_test,
    model=model
)

print("rmse ")
print(rmse)
print("mape")
print(mape)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15, 3])
ax.plot(pca_reduced_test['Appliances'], label='Original')
ax.plot(forecast, label='Forecast')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1, 1))
plt.show()





