import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader



def data_to_batch(train_data, test_data, val_data, add_info, batch_size, randomize=False):
    idx_train = list(range(train_data[0].shape[0]))

    if randomize:
        random.shuffle(idx_train)

        if add_info:
            train_dataset = DataLoader(
                TensorDataset(
                    torch.from_numpy(train_data[0][idx_train]).float(),
                    torch.from_numpy(train_data[1][idx_train, 0].astype(int)).long(),
                    torch.from_numpy(train_data[1][idx_train, 1:]).float(),
                ),
                batch_size=batch_size,
                shuffle=True,
            )

            val_dataset = DataLoader(
                TensorDataset(
                    torch.from_numpy(val_data[0]).float(),
                    torch.from_numpy(val_data[1][:, 0].astype(int)).long(),
                    torch.from_numpy(val_data[1][:, 1:]).float(),
                ),
                batch_size=batch_size,
                shuffle=True,
            )

        else:
            train_dataset = DataLoader(
                TensorDataset(
                    torch.from_numpy(train_data[0][idx_train]).float(),
                    torch.from_numpy(train_data[1][idx_train]).long(),
                ),
                batch_size=batch_size,
                shuffle=True,
            )

            val_dataset = DataLoader(
                TensorDataset(
                    torch.from_numpy(val_data[0]).float(),
                    torch.from_numpy(val_data[1]).long(),
                ),
                batch_size=batch_size,
                shuffle=True,
            )

        return train_dataset, 0, val_dataset

    if add_info:
        train_dataset = DataLoader(
            TensorDataset(
                torch.from_numpy(train_data[0][idx_train]).float(),
                torch.from_numpy(train_data[1][idx_train, 0].astype(int)).long(),
                torch.from_numpy(train_data[1][idx_train, 1:]).float(),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        test_dataset = DataLoader(
            TensorDataset(
                torch.from_numpy(test_data[0]).float(),
                torch.from_numpy(test_data[1][:, 0].astype(int)).long(),
                torch.from_numpy(test_data[1][:, 1:]).float(),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        val_dataset = DataLoader(
            TensorDataset(
                torch.from_numpy(val_data[0]).float(),
                torch.from_numpy(val_data[1][:, 0].astype(int)).long(),
                torch.from_numpy(val_data[1][:, 1:]).float(),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

    else:
        train_dataset = DataLoader(
            TensorDataset(
                torch.from_numpy(train_data[0][idx_train]).float(),
                torch.from_numpy(train_data[1][idx_train]).long(),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        test_dataset = DataLoader(
            TensorDataset(
                torch.from_numpy(test_data[0]).float(),
                torch.from_numpy(test_data[1]).long(),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        val_dataset = DataLoader(
            TensorDataset(
                torch.from_numpy(val_data[0]).float(),
                torch.from_numpy(val_data[1]).long(),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

    return train_dataset, test_dataset, val_dataset