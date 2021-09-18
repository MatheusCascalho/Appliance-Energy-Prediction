from sklearn.decomposition import PCA
from pyFTS.common.Transformations import Transformation
import pandas as pd

class PcaTransformation(Transformation):
    def __init__(self,n_components):
        self.n_components = n_components
        self.pca = PCA(n_components = self.n_components)
        self.is_multivariate = True


    def apply(self, data, param=None, **kwargs):
        endogen_variable = kwargs.get('endogen_variable', None)
        #names = kwargs.get('names', ('x', 'y'))

        names = []
        for i in range(0, self.n_components):
            names.append('C' + str(i))

        if endogen_variable not in data.columns:
            endogen_variable = None
        cols = data.columns[:-1] if endogen_variable is None else [col for col in data.columns if
                                                                   col != endogen_variable]

        self.pca.fit(data[cols].values)
        transformed = self.pca.transform(data[cols])
        new = pd.DataFrame(transformed, columns=list(names))
        new[endogen_variable] = data[endogen_variable].values
        return new