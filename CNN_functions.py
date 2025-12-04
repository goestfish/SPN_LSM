import random
import psutil
import GPUtil

import tensorflow as tf
import numpy as np
import time
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score
from sklearn import metrics as sk_metrics
from sklearn.metrics import balanced_accuracy_score
import gc
#from VAE import Resnet_VAE, make_pretrain_encoder
from VAE import make_pretrain_encoder
from utils import data_to_batch
import sys
tf.random.set_seed(0)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print('fix GPU growth')
    except RuntimeError as e:
        print(e)

def get_memory_usage():
    """Returns the current CPU and GPU memory usage in GB."""
    # Get CPU memory usage
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / (1024 ** 3)  # Convert bytes to GB

    # Get GPU memory usage (sum over all available GPUs)
    gpus = GPUtil.getGPUs()
    gpu_memory = sum(gpu.memoryUsed for gpu in gpus) if gpus else 0  # in GB

    return cpu_memory, gpu_memory

class Discretizer_layer(tf.keras.layers.Layer):
    def __init__(self,initial_value=1.0,dtype=tf.dtypes.float32,input_shape=16):
        super(Discretizer_layer,self).__init__()
        # with tf.name_scope("%s_%s" % (node.__class__.__name__, node.id)):
        self.bin = tf.Variable(initial_value=[initial_value]*input_shape, dtype=dtype, trainable=True,name='bin')  # name="mean",
       # self.discretizer=tf.keras.layers.Discretization(bin_boundaries=[self.bin],dtype=dtype)

    def call(self, inputs):
        #return tf.searchsorted(self.bin, inputs, side='right')#self.discretizer(inputs)
        return tf.where(inputs > self.bin, 1.0, 0.0)
def create_CNN(grid_params,input_shape,num_classes,add_info):
    use_augmentation=False
    use_rescaling=False
    num_layer=grid_params.num_layer
    dropout=grid_params.dropout
    filter_size=grid_params.filter_size
    last_num_filters=grid_params.last_num_filters

    last_pic_divisor = 2 ** (num_layer - 1)


    last_pic_len = int(input_shape[0]/last_pic_divisor)


    layer_arr = [tf.keras.Input(shape=input_shape,dtype=tf.dtypes.float32)]
    if use_augmentation:
        data_augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal", input_shape=input_shape),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ]
        )
        layer_arr.append(data_augmentation)
    if use_rescaling:
        layer_arr.append(layers.Rescaling(1. / 255))
    for lay_num in range(num_layer-1):
        num_filters=16 * (2 ** lay_num)
        layer_arr.extend([

            layers.Dropout(dropout / 2 ** lay_num),

            layers.Conv2D(num_filters, filter_size, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            tf.keras.layers.BatchNormalization(),
        ])
    layer_arr.extend([
        layers.Dropout(dropout / 2 ** num_layer),
        layers.Conv2D(last_num_filters, filter_size, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),

    ])
    layer_arr.extend([
        layers.MaxPooling2D(pool_size=(last_pic_len,last_pic_len)),
        layers.Flatten(),
        layers.Dense(num_classes, name="outputs")
    ])
    if num_classes==1:
        layer_arr.append(layers.Sigmoid())

    model = Sequential(layer_arr)
    print('num trainable vars',len(model.trainable_variables))

    return model



def load_CNN(grid_params, input_shape, num_classes,debugging,ckpt,manager,checkpoint_path,
             train_data,test_data,val_data,add_info):
    if grid_params.load_pretrain_model:
        model=load_pretrain_costume(grid_params,input_shape,num_classes)
    else:
        model = create_CNN(grid_params, input_shape, num_classes,add_info)
    learning_rate = grid_params.fine_tune_rate#grid_params.learning_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  # reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
                  metrics=['accuracy'])

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    if debugging:
        print('DEBUGGING show example CNN LOAD weights',model.trainable_variables[0].numpy()[0,0,0,:5])
    verbose=0


    if add_info:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0], train_data[1][:,0].astype(int))).shuffle(10).batch(grid_params.batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data[0], test_data[1][:,0].astype(int))).shuffle(10).batch(grid_params.batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data[0], val_data[1][:,0].astype(int))).shuffle(10).batch(grid_params.batch_size)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0], train_data[1])).shuffle(10).batch(grid_params.batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data[0], test_data[1])).shuffle(10).batch(grid_params.batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data[0], val_data[1])).shuffle(10).batch(grid_params.batch_size)



    train_loss, train_acc = model.evaluate(train_dataset, verbose=verbose)
    test_loss, test_acc = model.evaluate(test_dataset, verbose=verbose)
    val_loss, val_acc = model.evaluate(val_dataset, verbose=verbose)
    print('CNN train entropy',train_loss,'train acc', train_acc,'test entropy',
          test_loss, 'test acc',test_acc,flush=True)

    return model,ckpt,manager,[[train_loss, train_acc],[val_loss, val_acc],[test_loss, test_acc]]

def load_VAE(params,add_info,num_classes,input_shape,checkpoint_path,path=''):
    model = Resnet_VAE(
        filter_size=params.filter_size,
        num_layer=params.num_layer,
        input_shape=input_shape,
        batchnorm_integration=params.batchnorm_integration,
        shortcut=params.shortcut,
        activation=params.activation,
        num_filter_encoder=params.num_filter_encoder,
        strides_encoder=params.strides_encoder,
        num_filter_decoder=params.num_filter_decoder,
        strides_decoder=params.strides_decoder,
        latent_dim=params.latent_dim,
        end_dim_enc=params.end_dim_enc,
        learning_rate=params.learning_rate,
        semi_supervised=params.semi_supervised,
        num_classes=num_classes,
        dropout=params.dropout,
        load_pretrain_model=params.load_pretrain_model,
        add_info=add_info,
        loss_weights=params.loss_weights,
        VAE_fine_tune=params.VAE_fine_tune,
        path=path,
        use_KLD_anneal=params.use_KLD_anneal)

    # print model summary
    #model.summary()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=model.optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)
    #model.summary()
    #restored=0
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    #    restored=1
    return model



def load_VAE_and_eval(params, input_shape, num_classes,debugging,ckpt,manager,checkpoint_path,
             train_dataset,test_dataset,val_dataset,add_info):
    model=load_VAE(params,add_info,num_classes,input_shape,checkpoint_path)
    #else:
    #    print("ERROR: No Checkpoint file available!", checkpoint_path)
    if debugging:
        print('DEBUGGING show example CNN LOAD weights',model.trainable_variables[0].numpy()[0,0,0,:5])
    verbose=0



    train_losses = model.evaluate_(train_dataset, verbose=verbose)
    test_losses = model.evaluate_(test_dataset, verbose=verbose)
    val_losses = model.evaluate_(val_dataset, verbose=verbose)
    #print('CNN train entropy',train_loss,'train acc', train_acc 'test acc',test_acc,flush=True)




    return model,ckpt,manager,[[train_losses],[val_losses],[test_losses]]




def load_pretrain_costume(grid_params,input_shape,num_classes):
    base_model = tf.keras.models.load_model('efficientnetb7_saved_model')

    print('STOP-----')
    # freeze
    print('num layers in base model', len(base_model.layers))
    # for layer in base_model.layers:
    #    layer.trainable = False

    layer_name = 'block4a_activation'  # 'block7a_project_conv'
    partial_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

    # add input layer with gaussian noise:
    net_input = tf.keras.Input(shape=input_shape, dtype=tf.dtypes.float32)

    # x= tf.keras.layers.GaussianNoise(0.05)(net_input)
    x = partial_model(net_input)

    # x = base_model.get_layer(layer_name).output
    # Add new dropout layer with a custom rate
    x = layers.Dropout(grid_params.dropout)(x)  # Adjust dropout rate as needed #grid_params.dropout
    # Add conv layer to reduce filter size:
    #print('DEBUG CNN CONSTRUCTION',input_shape,grid_params.latent_dim, grid_params.filter_size,x)
    x = layers.Conv2D(grid_params.latent_dim, grid_params.filter_size, padding='same', activation='relu')(x)#tanh
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    next_dropout=max(0.0,grid_params.dropout-0.1)
    x = layers.Dropout(next_dropout)(x)  # Adjust dropout rate as needed #grid_params.dropout / 2
    #x = tf.keras.layers.GaussianNoise(0.1)(x)

    x = layers.MaxPooling2D(pool_size=(8, 8))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(num_classes, name="outputs")(x)

    model = tf.keras.models.Model(inputs=net_input, outputs=x)
    return model


def train_embedding_cnn(grid_params, val_data,input_shape,
                        num_classes,checkpoint_path,load_data,split,train_data,add_info,acc_stop=True):

    train_start=time.time()
    epochs=grid_params.epochs
    learning_rate=grid_params.learning_rate
    verbose = 2#1#2
    if add_info:
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (val_data[0], val_data[1][:, 0].astype(int))).shuffle(10).batch(grid_params.batch_size)

    if grid_params.load_pretrain_model:
        #model=load_pretrain(grid_params,input_shape,num_classes,split)
        model=load_pretrain_costume(grid_params,input_shape,num_classes)
        print('learningrate:',learning_rate)


        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)#tf.keras.optimizers.Adamax(learning_rate = 0.001) #
    else:

        model=create_CNN(grid_params,input_shape,num_classes,add_info)


        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if num_classes==1:
        loss=tf.keras.losses.MeanSquaredError()
        metrics = ['MAE']
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        #loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']

    model.compile(optimizer=optimizer,loss=loss, metrics=metrics)
    # Todo if path exists - delete model! - do not use old model - critical for gridsearch
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)



    if acc_stop:
        best_val_loss=0
    else:
        best_val_loss = float('inf')
    if load_data:
        ckpt.restore(manager.latest_checkpoint)
    else:

        val_loss = model.evaluate(val_dataset, verbose=0,)[0]
        print('CNN val loss:', val_loss)

        for epoch in range(epochs):
            start_time_train=time.time()
            # shuffle dataset:
            idx_train = list(range(train_data[0].shape[0]))
            random.shuffle(idx_train)
            X=train_data[0][idx_train]
            if add_info:
                y = train_data[1][idx_train,0]
                other = train_data[1][idx_train,1:]
            else:
                y = train_data[1][idx_train]


            train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(grid_params.batch_size)
            model.fit(train_dataset, epochs=1,
                            #validation_data=val_dataset,
                            verbose=verbose)
            if not epoch:
                #model.layers[0].summary()
                model.summary()

            [val_loss,acc_loss] = model.evaluate(val_dataset, verbose=2)
            improve=False
            if acc_stop:
                if acc_loss>best_val_loss:
                    best_val_loss=acc_loss
                    ckpt.step.assign_add(1)
                    manager.save()
                    improve=True
                    print('INITIAL weights', model.trainable_variables[0].numpy()[0, 0, 0, :5])

            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save the model weights
                    ckpt.step.assign_add(1)
                    manager.save()
                    improve=True
                    print('INITIAL weights', model.trainable_variables[0].numpy()[0, 0, 0, :5])
            print('CNN val loss:', val_loss,acc_loss,'improve',improve,'time:',(time.time()-start_time_train)/60)


    return model,manager,ckpt


def validate_data(val_dataset,add_info,model,beta=0):
    losses = np.zeros((4 + 2))
    num_rounds = 0
    predictions = []
    all_z = []
    gt = []
    for val_rec in val_dataset:
        if add_info:
            test_x, test_y, test_other = val_rec


        else:
            test_other = None
            test_x, test_y = val_rec
        loss, rec_loss, kld, clf, other_losses, pred, z, _ = model.execute_net_xy(test_x, test_y,test_other, training=False,beta=beta)
        losses[0] += loss.numpy()
        losses[1] += rec_loss.numpy()
        losses[2] += kld.numpy()
        losses[3] += clf.numpy()
        predictions.extend(pred.numpy().tolist())
        gt.extend(test_y.numpy().tolist())  # .numpy()
        for i, tmp_loss in enumerate(other_losses):
            losses[4 + i] += tf.reduce_mean(tmp_loss)
        all_z.append([z.numpy(), test_y.numpy().tolist()])

        num_rounds += 1
    return all_z,predictions,gt,losses,num_rounds

def beta_scheduler(epoch, total_epochs,epoch_steps=100, max_beta=1.0):
    # Linear annealing
    return min(max_beta, (epoch%epoch_steps) / epoch_steps)

def load_model(params,num_classes,add_info,checkpoint_path,checkpoint_path_tmp,le_warmup,input_shape,encoder,init=True):
    print(input_shape)
    model = Resnet_VAE(
        filter_size=params.filter_size,
        num_layer=params.num_layer,
        input_shape=input_shape,
        batchnorm_integration=params.batchnorm_integration,
        shortcut=params.shortcut,
        activation=params.activation,
        num_filter_encoder=params.num_filter_encoder,
        strides_encoder=params.strides_encoder,
        num_filter_decoder=params.num_filter_decoder,
        strides_decoder=params.strides_decoder,
        latent_dim=params.latent_dim,
        end_dim_enc=params.end_dim_enc,
        learning_rate=params.learning_rate,
        semi_supervised=params.semi_supervised,
        num_classes=num_classes,
        dropout=params.dropout,
        load_pretrain_model=params.load_pretrain_model,
        add_info=add_info,
        loss_weights=params.loss_weights,
        VAE_fine_tune=params.VAE_fine_tune,
        use_GAN=params.GAN,
        use_KLD_anneal=params.use_KLD_anneal,
        le_warmup=le_warmup,gauss_std=params.gauss_std,encoder=encoder)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=model.optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

    ckpt_tmp = tf.train.Checkpoint(step=tf.Variable(1), optimizer=model.optimizer, net=model)
    manager_tmp = tf.train.CheckpointManager(ckpt_tmp, checkpoint_path_tmp, max_to_keep=1)




    if init:
        ckpt.restore(manager.latest_checkpoint)
        ckpt_tmp.step.assign_add(1)
        manager_tmp.save()
    else:
        ckpt_tmp.restore(manager_tmp.latest_checkpoint)


    return model,manager, ckpt,manager_tmp, ckpt_tmp

def train_embedding_VAE(params, train_data,test_data,val_data, input_shape,
                        num_classes, checkpoint_path, load_data, split,add_info,checkpoint_path_tmp,acc_stop=True):
    epochs = params.epochs
    encoder=None
    if params.load_pretrain_model:
        encoder=make_pretrain_encoder(params.filter_size, num_layer=0, input_shape=input_shape, batchnorm_integration=params.batchnorm_integration, num_filter=0, shortcut=0, strides=0,
                              activation=0, encoder_name=0, dtype=0, dilations=0, dropout=params.dropout, end_dim=params.end_dim_enc, path='')

    model,manager, ckpt,manager_tmp, ckpt_tmp=load_model(params,num_classes,add_info,checkpoint_path,checkpoint_path_tmp,params.le_warmup,input_shape,encoder[0],init=True)


    train_dataset, test_dataset, val_dataset = data_to_batch(train_data, test_data, val_data, add_info, params.batch_size)

    load_data=0
    all_val_losses=[]
    all_debugging_stuff=[]
    if acc_stop:
        best_val_loss = 0
    else:
        print('stop for best loss on L')
        best_val_loss = float('inf')
    if load_data:
        #ckpt.restore(manager.latest_checkpoint)
        print('I used an old Model')
    else:

        val_loss = model.evaluate_(val_dataset, verbose=0, )[0]
        print('CNN val loss:', val_loss)

        for epoch in range(epochs):

            #print('Weights before clear session', epoch, model.trainable_variables[0].numpy()[0, 0, 0, :5])
            print('delete',flush=True)
            del train_dataset,val_dataset
            train_dataset = None
            val_dataset = None

            gc.collect()  # Python garbage collection
            tf.keras.backend.clear_session()  # Clears the global Keras session
            gc.collect()

            train_dataset, _, val_dataset = data_to_batch(train_data, test_data, val_data, add_info,params.batch_size,randomize=True)
            print('load finished',flush=True)
            #if epoch and epoch % 10 == 0:
            warmup=0#1 if epoch==0 and params.le_warmup else 0

            cpu_mem_before, gpu_mem_before = get_memory_usage()

            print(
                f"Epoch {epoch + 1}/{epochs} -| CPU: {cpu_mem_before:.2f} GB | GPU: {gpu_mem_before:.2f} GB",
                flush=True)

            beta = beta_scheduler(epoch, epochs)
            start_time_train = time.time()

            if model.semi_supervised:


                loss, rec_loss, kld, clf_loss, mae, mse = model.train_semi(train_dataset,beta=beta)
                train_loss=[loss, rec_loss, kld, clf_loss, mae, mse]
                print("Train Semi | Epoch: {:03d} | Loss: {:.4f} | Rec Loss: {:.4f} | "
                      "KLD: {:.4f} | Clf Loss: {:.4f} | MAE: {:.4f} | MSE: {:.4f}".format(
                    epoch, loss, rec_loss, kld, clf_loss, mae, mse),flush=True)

                # every 3 epoch do validation:

                # test if loss is nan:
                if np.isnan(loss):
                    print('load old checkpoint')
                    #ckpt.restore(manager.latest_checkpoint)
                    ckpt.restore(manager_tmp.latest_checkpoint)
                    print('Loaded weights for epoch', epoch, model.trainable_variables[0].numpy()[0, 0, 0, :5])



                if epoch%10==0 and not np.isnan(loss):
                    print('start eval',flush=True)

                    all_z,predictions,gt,losses,num_rounds=validate_data(val_dataset,add_info,model,beta=beta)
                    all_z_train, predictions_train, gt_train, losses_train, num_rounds_train = validate_data(train_dataset, add_info, model,beta=beta)


                    # calculate accuracy:
                    print('pred shapes',len(predictions),np.asarray(predictions).shape)
                    softmax_pred=tf.nn.softmax(predictions,axis=-1)
                    arg_max = np.argmax(softmax_pred, axis=-1)
                    acc_loss = accuracy_score(gt, arg_max, normalize=True)
                    balanced_acc = balanced_accuracy_score(gt, arg_max)
                    losses=losses / num_rounds
                    val_loss =losses[0]
                    losses = np.round(losses, decimals=3)
                    rec=0
                    if params.VAE_debug:
                        test_x = test_data[0][:16]
                        train_x=train_data[0][:16]
                        if add_info:
                            test_y = test_data[1][:16, 0]
                            train_y=train_data[1][:16, 0]
                            test_add_info=test_data[1][:16, 1:]
                            train_add_info=train_data[1][:16, 1:]
                        else:
                            test_y = test_data[1][:16]
                            train_y = train_data[1][:16]
                            test_add_info=None
                            train_add_info=None


                        ##########################################################################

                        # z , reconstruction of val dataset (no random stuff)
                        out_rec = model.execute_net_xy(test_x, test_y,test_add_info, training=False)
                        out_rec_train = model.execute_net_xy(train_x, train_y,train_add_info, training=False)
                        z,gt=zip(*all_z)
                        z=np.concatenate(z,axis=0)
                        gt=np.concatenate(gt,axis=0)

                        z_train,gt_train=zip(*all_z_train)
                        z_train=np.concatenate(z_train,axis=0)
                        gt_train=np.concatenate(gt_train,axis=0)

                        rec=out_rec[-1].numpy()
                        all_debugging_stuff.append([[losses,z,gt,rec,predictions],
                                                    [train_loss,z_train,gt_train,out_rec_train[-1].numpy(),predictions_train]])
                        size_in_bytes = sys.getsizeof(all_debugging_stuff)
                        size_in_mb = size_in_bytes / (1024 ** 2)  # Convert to MB
                        print(f"Memory used by all_debugging_stuff: {size_in_mb:.2f} MB")
                        del test_x,train_x,test_y,train_y

                    print(
                        "Val Results Semi | Epoch: {:03d} | Acc Loss: {:.4f} | Balanced Acc: {:.4f} | MAE: {:.4f}|min {:.4f}|max{:.4f}".format(
                            epoch, acc_loss, balanced_acc, losses[-2],np.min(rec),np.max(rec)), flush=True)

                    improve = False
                    if acc_stop:
                        if balanced_acc > best_val_loss:
                            best_val_loss = balanced_acc
                            ckpt.step.assign_add(1)
                            manager.save()
                            improve = True
                            print('SAVE weights', model.trainable_variables[0].numpy()[0, 0, 0, :5],flush=True)

                    else:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            # Save the model weights
                            ckpt.step.assign_add(1)
                            manager.save()
                            improve = True
                            print('SAVE weights', model.trainable_variables[0].numpy()[0, 0, 0, :5],flush=True)
                    print(
                        "CNN Val Loss: {:.4f} | Acc Loss: {:.4f} | Improve: {} | Time: {:.2f} min | Beta: {:.4f}".format(
                            val_loss, acc_loss, improve, (time.time() - start_time_train) / 60, beta),flush=True)

                    all_val_losses.append(losses)
            else:
                loss, rec_loss, kld, mae, mse = model.train(train_dataset)
                print(
                    "Train | Epoch: {:03d} | Loss: {:.4f} | Rec Loss: {:.4f} | KLD: {:.4f} | MAE: {:.4f} | MSE: {:.4f}".format(
                        epoch, loss, rec_loss, kld, mae, mse))

                end_time = time.time()
                losses = np.zeros((3 + 2))
                num_rounds = 0
                for test_x, y in val_dataset:
                    loss, rec_loss, kld, other_losses = model.execute_net(test_x, training=False)
                    losses[0] += loss
                    losses[1] += rec_loss
                    losses[2] += kld
                    for i, tmp_loss in enumerate(other_losses):
                        losses[3 + i] += tf.reduce_mean(tmp_loss)

                    num_rounds += 1
                losses=losses / num_rounds
                val_loss=losses[0]
                losses = np.round(losses / num_rounds, decimals=3)

                print('t_results', epoch, losses, flush=True)
                improve=False
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save the model weights
                    ckpt.step.assign_add(1)
                    manager.save()
                    improve = True
                    print('INITIAL weights', model.trainable_variables[0].numpy()[0, 0, 0, :5])
                print('CNN val loss:', val_loss, 'improve', improve, 'time:',
                      (time.time() - start_time_train) / 60,flush=True)
                all_val_losses.append(losses)


            ckpt_tmp.step.assign_add(1)
            manager_tmp.save()
            #print('Weights of tmp session', epoch, model.trainable_variables[0].numpy()[0, 0, 0, :5])
            print('time',(time.time() - start_time_train) / 60,flush=True)



        ############################################################################


    return model, manager, ckpt,all_val_losses,all_debugging_stuff


def get_layer_embeddings(grid_params,model, layer_names, all_datasets,get_max=True,add_info=False):
    print('START EMBEDDING',layer_names)
    if grid_params.use_VAE:
        intermediate_layer_model=model.clf_model()
    else:
        layer_output = [model.get_layer(layer_name).output for layer_name in layer_names]

        intermediate_layer_model = tf.keras.Model(inputs=model.input,outputs=layer_output)

    output_embeds = []
    all_labels=[[] for _ in range(len(all_datasets))]
    all_other_info = [[] for _ in range(len(all_datasets))]
    num_embedding_layers=1
    for ds_idx,input_dataset in enumerate(all_datasets):
        all_batch_embeddings = [[] for _ in range(num_embedding_layers)]
        for ds_record in input_dataset:
            if add_info:
                images, labels,other = ds_record
                all_other_info[ds_idx].append(other.numpy())
                #print('img shape',images.shape)
            else:
                images, labels=ds_record

            prediction_0 = intermediate_layer_model.predict(images,verbose=0)
            if grid_params.use_VAE:
                prediction_0=prediction_0[0]

            all_labels[ds_idx].append(labels.numpy())

            if len(layer_names) > 1:
                for lay in range(num_embedding_layers):
                    ##############################
                    # TODO avg_pooling layer?!
                    if get_max:
                        diff = np.max(prediction_0[lay], axis=(1, 2))
                    else:
                        diff=prediction_0[lay]
                    all_batch_embeddings[lay].extend(diff)
            else:
                if get_max:
                    diff = np.max(prediction_0, axis=(1, 2))
                else:
                    diff=prediction_0
                all_batch_embeddings[0].extend(diff)
        all_labels[ds_idx]=np.concatenate(all_labels[ds_idx],axis=0)
        if add_info:
            all_other_info[ds_idx] = np.concatenate(all_other_info[ds_idx], axis=0)

        curr_embeds = [np.stack(entry, axis=0) for entry in all_batch_embeddings]
        if len(curr_embeds) > 1:
            concat_embeds = np.concatenate(curr_embeds, axis=-1)
            output_embeds.append(concat_embeds)
        else:
            output_embeds.append(curr_embeds[0])
            print(curr_embeds[0].shape)

    return output_embeds,all_labels,all_other_info
