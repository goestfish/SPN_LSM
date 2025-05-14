import numpy as np
import time
from spn.algorithms.LearningWrappers import learn_parametric
from spn.algorithms.MPE import mpe
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context,Sum, assign_ids
from sklearn.metrics import accuracy_score




def learn_classifier(data, debugging,ds_context, spn_learn_wrapper, label_idx, **kwargs):
    spn = Sum()
    label_ids=[]
    for label, count in zip(*np.unique(data[:, label_idx], return_counts=True)):
        branch = spn_learn_wrapper(data[data[:, label_idx] == label, :], ds_context, **kwargs)
        spn.children.append(branch)
        spn.weights.append(count / data.shape[0])
        label_ids.append(label)

    spn.scope.extend(branch.scope)
    assign_ids(spn)


    return spn,label_ids

def create_SPN(param_grid,train_data,test_data,test_y,input_dim,debugging):
    train_start=time.time()
    parametric_types=[Categorical]+[Gaussian for _ in range(input_dim)]
    min_instances_slice=int(param_grid.min_instances_slice_percentage*train_data.shape[0])
    print('min_instances_slice',min_instances_slice,train_data.shape[0])
    spn_classification,label_ids = learn_classifier(train_data,debugging,
                                          Context(parametric_types=parametric_types).add_domains(
                                              train_data),
                                          learn_parametric, 0,
                                          min_instances_slice=min_instances_slice,
                                          min_features_slice=param_grid.min_features_slice,
                                          multivariate_leaf=False,
                                          leaves=None,
                                          n_clusters=param_grid.n_clusters,
                                          cols=param_grid.col_split,
                                          rows=param_grid.row_split
                                          )

    prediction= mpe(spn_classification, test_data)[:, 0]
    acc=accuracy_score(test_y, prediction, normalize=True)
    #print('mpe example',list(zip(prediction,test_y))[:100],flush=True)
    print('TEST Result Acc after SPN training', acc,'time:',(time.time()-train_start)//60,flush=True)

    return spn_classification,label_ids







