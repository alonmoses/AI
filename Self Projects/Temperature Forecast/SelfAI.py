import torch
import numpy as np

def define_type(data_frame, col_list, type_wanted):
    for col in col_list:
        data_frame[col] = data_frame[col].astype(type_wanted)
    return data_frame

def stack_cols(data_frame, col_list, tensor_type, cat=0):
    if cat == 1:
        stacked_list = np.stack([data_frame[col].cat.codes.values for col in col_list], axis=1)
    else:
        stacked_list = np.stack([data_frame[col] for col in col_list], axis=1)

    stacked_list = torch.tensor(stacked_list, dtype=tensor_type)
    return stacked_list

def create_sets(cont_stack, cat_stack, labels, batchSize):
    sets_list = [cont_stack, cat_stack, labels]
    splitted_list = []
    testSize = int(batchSize * 0.2)
    for sets in sets_list:
        splitted_list.append(sets[:batchSize-testSize])
        splitted_list.append(sets[batchSize-testSize:batchSize])
    return splitted_list