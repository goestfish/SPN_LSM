import tensorflow as tf
import random




def data_to_batch(train_data,test_data,val_data,add_info,batch_size,randomize=False):
    idx_train = list(range(train_data[0].shape[0]))

    if randomize:
        random.shuffle(idx_train)
        if add_info:
            # print(val_data[0].shape,val_data[1].shape,'val data shape')
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0][idx_train],
                                                                train_data[1][idx_train, 0].astype(int),
                                                                train_data[1][idx_train, 1:])).shuffle(10).batch(
                batch_size).cache().prefetch(tf.data.AUTOTUNE)
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (val_data[0], val_data[1][:, 0].astype(int), val_data[1][:, 1:])).shuffle(10).batch(
                batch_size).cache().prefetch(tf.data.AUTOTUNE)
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_data[0][idx_train], train_data[1][idx_train])).shuffle(10).batch(batch_size)
            val_dataset = tf.data.Dataset.from_tensor_slices((val_data[0], val_data[1])).shuffle(10).batch(batch_size)


        return train_dataset,0,val_dataset

    if add_info:
        #print(val_data[0].shape,val_data[1].shape,'val data shape')
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0][idx_train], train_data[1][idx_train,0].astype(int),train_data[1][idx_train,1:])).shuffle(10).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data[0], test_data[1][:,0].astype(int),test_data[1][:,1:])).shuffle(10).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data[0], val_data[1][:,0].astype(int),val_data[1][:,1:])).shuffle(10).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0][idx_train], train_data[1][idx_train])).shuffle(10).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data[0], test_data[1])).shuffle(10).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data[0], val_data[1])).shuffle(10).batch(batch_size)



    return train_dataset, test_dataset, val_dataset