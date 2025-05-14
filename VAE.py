import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics




def enc_layer(net,stride,filter_num,batchnorm_integration,shortcut,
              dilations,filter_size,trainable,dtype,old_h, old_num_filter,dropout,activation,
              name,gauss_std):

    convolution = tf.keras.layers.Conv2D(filter_num, filter_size, dilation_rate=dilations, strides=stride, padding="same",
                                  data_format='channels_last', use_bias=False, activation=None, name=name+'conv',trainable=trainable,dtype=dtype)(net)
    #convolution = tf.squeeze(convolution, axis=-2)

    if shortcut:

        shortcut = tf.keras.layers.AveragePooling2D(pool_size=filter_size, strides=stride, padding="same",
                                                  data_format="channels_last",trainable=trainable)(net)
        # reduce shortcut:
        #shortcut = tf.squeeze(shortcut, axis=-1)

        shortcut= tf.keras.layers.Dense(1, activation=None, use_bias=True, input_shape=(-1, old_h,old_h, old_num_filter),trainable=trainable,dtype=dtype)(shortcut)
        convolution = tf.keras.layers.Concatenate(axis=-1)([convolution, shortcut])
    if batchnorm_integration:
        convolution = tf.keras.layers.BatchNormalization(trainable=trainable,dtype=dtype)(convolution)

    if dropout:
        convolution = tf.keras.layers.Dropout(rate=dropout)(convolution)

    convolution = tf.keras.layers.GaussianNoise(gauss_std)(convolution)
    convolution = activation(convolution)

    return convolution
def dec_layer(net,stride,filter_num,batchnorm_integration,shortcut,
              dilations,filter_size,trainable,dtype,old_h, old_num_filter,dropout,activation,
              name,gauss_std=0.0):

    net = tf.keras.backend.resize_images(net, height_factor=stride, width_factor=stride,data_format="channels_last",interpolation="nearest")
    convolution = tf.keras.layers.Conv2D(filter_num, filter_size, dilation_rate=dilations, strides=1, padding="same",
                                  data_format='channels_last', use_bias=False, activation=None)(net)
    if shortcut:
        shortcut = tf.keras.layers.Dense(1, activation=None, use_bias=True)(net)
        convolution = tf.keras.layers.Concatenate(axis=-1)([convolution, shortcut])
    if batchnorm_integration:
        convolution = tf.keras.layers.BatchNormalization()(convolution)
    net = tf.keras.layers.GaussianNoise(gauss_std)(net)
    convolution = activation(convolution)
    return convolution


def make_costume_encoder(filter_size, num_layer, input_shape, batchnorm_integration, num_filter, shortcut, strides,
                         activation, encoder_name, dtype, dilations, dropout, end_dim,gauss_std=0.0):
    inputs = tf.keras.Input(shape=input_shape, dtype=dtype)
    net = inputs  # /255.0
    img_size = input_shape[0]
    num_filter = [input_shape[2]] + num_filter
    for layer_num, stride in enumerate(strides):
        old_num_filter = num_filter[layer_num]
        if layer_num:
            old_num_filter += 1
        net = enc_layer(net, stride, num_filter[layer_num + 1], batchnorm_integration, shortcut, dilations, filter_size,
                        trainable=True, dtype=dtype,
                        old_h=img_size,
                        old_num_filter=num_filter[layer_num],
                        dropout=dropout,
                        activation=activation,
                        name=encoder_name + '_' + str(layer_num),gauss_std=gauss_std)
        img_size /= stride

    # net = tf.keras.layers.MaxPooling2D(pool_size=(img_size, img_size))(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(end_dim, activation=None)(net)
    net = activation(net)

    model = tf.keras.Model(inputs=inputs, outputs=net)
    return model, int(img_size)


def make_pretrain_encoder(filter_size, num_layer, input_shape, batchnorm_integration, num_filter, shortcut, strides,
                          activation, encoder_name, dtype, dilations, dropout, end_dim,path=''):


    base_model = tf.keras.models.load_model(path+'efficientnetb7_saved_model')#(path+'efficientnetb7_saved_model_128.keras')#
    layer_name = 'block4a_activation'  # 'block7a_project_conv'
    partial_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

    # add input layer with gaussian noise:
    net_input = tf.keras.Input(shape=input_shape, dtype=tf.dtypes.float32)
    # x= tf.keras.layers.GaussianNoise(0.05)(net_input)
    x = partial_model(net_input)

    # x = base_model.get_layer(layer_name).output
    # Add new dropout layer with a custom rate
    x = tf.keras.layers.Dropout(dropout)(x)  # Adjust dropout rate as needed #grid_params.dropout
    # Add conv layer to reduce filter size:

    x = tf.keras.layers.Conv2D(end_dim, filter_size, padding='same', activation='relu')(x)#tanh
    if batchnorm_integration:
        x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

    next_dropout=max(0.0,dropout-0.1)
    x = tf.keras.layers.Dropout(next_dropout)(x)  # Adjust dropout rate as needed #grid_params.dropout / 2
    #x = tf.keras.layers.GaussianNoise(0.1)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(8, 8))(x)
    x = tf.keras.layers.Flatten()(x)


    model = tf.keras.models.Model(inputs=net_input, outputs=x)


    return model, 8





def make_encoder(filter_size,num_layer,input_shape,
                 batchnorm_integration,
                 num_filter=[16,32,62,124,248,124,124,32],
                 shortcut=True,
                 strides=[2,1,2,1,2,1,2,1],
                 activation=tf.keras.activations.relu,
                 encoder_name='encoder',
                 dtype=tf.float32,
                 dilations=1,
                 dropout=0.0,
                 end_dim=128,load_pretrain_model=0,path='',gauss_std=0.0):
    if load_pretrain_model:
        return make_pretrain_encoder(filter_size,num_layer,input_shape,
                 batchnorm_integration,
                 num_filter=num_filter,
                 shortcut=shortcut,
                 strides=strides,
                 activation=activation,
                 encoder_name=encoder_name,
                 dtype=dtype,
                 dilations=dilations,
                 dropout=dropout,
                 end_dim=end_dim,path=path)#,gauss_std=gauss_std
    else:
        return make_costume_encoder(filter_size,num_layer,input_shape,
                 batchnorm_integration,
                 num_filter=num_filter,
                 shortcut=shortcut,
                 strides=strides,
                 activation=activation,
                 encoder_name=encoder_name,
                 dtype=dtype,
                 dilations=dilations,
                 dropout=dropout,
                 end_dim=end_dim,gauss_std=gauss_std)




def make_mlp_clf(input_shape,activation,batchnorm_integration,dropout=0.0,num_classes=2,layer=1,name='clf'):
    inputs = tf.keras.Input(shape=input_shape, dtype=dtype)
    net=inputs

    for i in range(layer):
        if dropout:
            net=tf.keras.layers.Dropout(dropout/8)(net)
        if batchnorm_integration:
            net = tf.keras.layers.BatchNormalization()(net)

        #net=tf.keras.layers.Dense(int(input_shape[0]/(2**(i+1))), name=name+str(i))(net)
        print(input_shape[0],(2 ** (i + 1)),input_shape[0] / (2 ** (i + 1)),int(input_shape[0] / (2 ** (i + 1))))
        net = tf.keras.layers.Dense(input_shape[0] / (2 ** (i + 1)), name=name + str(i))(net)
        net=activation(net)
    if dropout:
        net = tf.keras.layers.Dropout(dropout/8)(net)
    if batchnorm_integration:
        net = tf.keras.layers.BatchNormalization()(net)

    net = tf.keras.layers.Dense(num_classes, name=name + str(layer))(net)

    model = tf.keras.Model(inputs=inputs, outputs=net)
    return model

def make_decoder(filter_size,num_layer,input_shape,output_shape,
                 batchnorm_integration,
                 num_filter=[1,32,62,124,248,124,124,124],
                 shortcut=True,
                 strides=[2,1,2,1,2,1,2,2],
                 activation=tf.keras.activations.relu,
                 im_size=14,
                 encoder_name='encoder',
                 dtype=tf.float32,
                 dilations=1,
                 dropout=0.0,semi_supervised=False,gauss_std=0.0):
    inputs = tf.keras.Input(shape=input_shape, dtype=dtype)

    net = tf.keras.layers.Dense(im_size * im_size * 128, activation='relu')(inputs) #* 128
    net = tf.keras.layers.Reshape((im_size, im_size, 128))(net)#, 128

    #im_size=input_shape#[0]
    #num_filter=[input_shape[3]]+num_filter
    old_num_filter=128
    for layer_num,stride in enumerate(strides):
        #'''
        if layer_num < len(strides) - 1:
            curr_shortcut=shortcut
        else:
            curr_shortcut=False
            gauss_std=0.0
            activation=tf.keras.activations.sigmoid
        net=dec_layer(net,stride,num_filter[-(layer_num+1)],batchnorm_integration,curr_shortcut,dilations,filter_size,
                      trainable=True,dtype=dtype,
                      old_h=im_size,
                      old_num_filter=old_num_filter+1,
                      dropout=dropout,
                      activation=activation,
                      name=encoder_name+'_'+str(layer_num),gauss_std=gauss_std)



    model = tf.keras.Model(inputs=inputs, outputs=net)
    return model
dtype=tf.float32




class WarmUpLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, target_lr, warmup_steps):
        """ target_lr: Final learning rate after warmup
            warmup_steps: Number of steps to reach target LR
        """
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """ Linear warm-up from 0 to target_lr over warmup_steps """
        if self.warmup_steps > 0:
            return self.target_lr * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
        return self.target_lr  # Default LR if no warm-up needed

class Resnet_VAE(tf.keras.Model):
    def __init__(self,
                 filter_size,
                 num_layer,
                 input_shape,
                 batchnorm_integration,
                 shortcut,
                 activation,
                 num_filter_encoder,
                 strides_encoder,
                 end_dim_enc,
                 num_filter_decoder,
                 strides_decoder,
                 latent_dim,
                 learning_rate,
                 semi_supervised,
                 num_classes,
                 dropout,
                 load_pretrain_model,
                 add_info,
                 loss_weights,
                 use_KLD_anneal,
                 VAE_fine_tune,path='',use_GAN=False,le_warmup=0,gauss_std=0.0,encoder=0):
        super(Resnet_VAE, self).__init__()
        #self.dtype=dtype
        self.latent_dim=latent_dim
        self.input_shape_=input_shape
        self.load_pretrain_model=load_pretrain_model
        self.add_info=add_info
        self.loss_weights=loss_weights
        self.VAE_fine_tune=VAE_fine_tune
        self.use_KLD_anneal=use_KLD_anneal
        print('loss_weights',self.loss_weights)

        if activation == 'relu':
            self.activation = tf.keras.layers.LeakyReLU()
        elif activation == 'tanh':
            self.activation = tf.keras.activations.tanh

        if load_pretrain_model:
            self.encoder, im_size=encoder,8
        else:
            self.encoder,im_size =make_encoder(filter_size=filter_size,
                                               num_layer=num_layer,
                                               input_shape=input_shape,
                                                batchnorm_integration=batchnorm_integration,
                                                num_filter = num_filter_encoder,
                                                shortcut = shortcut,
                                                strides = strides_encoder,
                                                activation = self.activation,
                                                encoder_name = 'encoder',
                                                dtype = tf.float32,
                                                dilations = 1,
                                                dropout = dropout,
                                                end_dim = end_dim_enc,
                                               load_pretrain_model=load_pretrain_model,path=path,gauss_std=gauss_std)

        self.z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')
        self.z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')


        self.semi_supervised=semi_supervised
        dec_inp_shape = (latent_dim,)
        if semi_supervised:
            cls_input_shape=(latent_dim,)
            if self.add_info:
                cls_input_shape = (latent_dim+3,)

            self.classifier=make_mlp_clf(input_shape=cls_input_shape,
                                         activation=self.activation,
                                         batchnorm_integration=batchnorm_integration,
                                         dropout=0.0,
                                         num_classes=2,layer=3)

            self.clf_loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            if self.VAE_fine_tune==0:
                dec_inp_shape=(latent_dim+num_classes,)
            else:
                dec_inp_shape = (latent_dim,)
        self.decoder = make_decoder(filter_size=filter_size,
                                    num_layer=num_layer,
                                    input_shape=dec_inp_shape,
                                    output_shape=input_shape[0],
                                     batchnorm_integration=batchnorm_integration,
                                     num_filter=num_filter_decoder,
                                     shortcut=shortcut,
                                     strides=strides_decoder,
                                     activation=tf.keras.activations.relu,
                                     im_size=im_size,
                                     encoder_name='decoder',
                                     dtype=tf.float32,
                                     dilations=1,
                                     dropout=dropout,
                                    semi_supervised=semi_supervised,gauss_std=gauss_std)

        if le_warmup:
            warmup_steps = 5 * 40  # Warmup for first 5 epochs
            target_lr = learning_rate#1e-3

            lr_schedule = WarmUpLearningRateSchedule(target_lr=target_lr, warmup_steps=warmup_steps)


        else:
            lr_schedule =learning_rate


        #self.optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)##
        self.loss_fkts=[tf.keras.losses.MAE,tf.keras.losses.MSE]




    def train(self,train_gen):
        losses=np.zeros((3+2))
        num_rounds=0
        for (X,y) in train_gen:
            loss, rec_loss, kld,other_loss=self.train_step(X)
            losses[0]+=loss
            losses[1]+=rec_loss
            losses[2]+=kld
            for i,tmp_loss in enumerate(other_loss):
                losses[3+i]+=tf.reduce_mean(tmp_loss)
            num_rounds+=1
            #print(num_rounds,np.round(losses/num_rounds,decimals=4))
        return losses/num_rounds


    def train_semi(self,train_gen,beta=0):
        losses=np.zeros((4+2))
        num_rounds=0
        if self.add_info:
            for (X,y,other_data) in train_gen:
                loss, rec_loss, kld,clf_loss,other_loss=self.train_step_semi_label(X,y,other_data,beta=beta)
                losses[0]+=loss.numpy()
                losses[1]+=rec_loss.numpy()
                losses[2]+=kld.numpy()
                losses[3] += clf_loss.numpy()
                for i,tmp_loss in enumerate(other_loss):
                    losses[4+i]+=tf.reduce_mean(tmp_loss).numpy()
                num_rounds+=1
                #print(num_rounds,np.round(losses/num_rounds,decimals=4))
        else:
            for (X,y) in train_gen:
                loss, rec_loss, kld,clf_loss,other_loss=self.train_step_semi_label(X,y,None,beta=beta)
                losses[0]+=loss.numpy()
                losses[1]+=rec_loss.numpy()
                losses[2]+=kld.numpy()
                losses[3] += clf_loss.numpy()
                for i,tmp_loss in enumerate(other_loss):
                    losses[4+i]+=tf.reduce_mean(tmp_loss).numpy()
                num_rounds+=1
                #print(num_rounds,np.round(losses/num_rounds,decimals=4))
        return losses/num_rounds



    @tf.function
    def train_step(self,X,beta=0):
        with tf.GradientTape() as tape:
            loss,rec_loss,kld,other_loss=self.execute_net(X,training=True)
            #loss = tf.reduce_mean(loss)

        train_vars = self.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return loss,rec_loss,kld,other_loss

    @tf.function
    def train_step_semi_label(self,X,y,add_data,beta=0):
        with tf.GradientTape() as tape:
            loss,rec_loss,kld,clf_loss,other_loss,_,_,_=self.execute_net_xy(X,y,add_data,training=True,beta=beta)
            #loss = tf.reduce_mean(loss)

        train_vars = self.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return loss,rec_loss,kld,clf_loss,other_loss


    def encode_input(self,X,training=True):
        encoding=self.encoder(X)
        cls_input_mean = self.z_mean(encoding, training=training)
        cls_input_var = self.z_log_var(encoding, training=training)
        exp0 = tf.exp(cls_input_var * .5)
        eps0 = tf.random.normal(shape=(cls_input_mean.shape[0], cls_input_mean.shape[1]), mean=0, stddev=1,dtype=self.dtype)
        cls_input0 = eps0 * exp0+ cls_input_mean
        return cls_input_mean,cls_input_var,cls_input0
    def enc_dec(self,X,training=True):
        cls_input_mean, cls_input_var, cls_input0 = self.encode_input(X, training=training)
        decoding = self.decoder(cls_input0)
        return cls_input_mean,cls_input_var,cls_input0,decoding

    def embedding(self,X, training=False):
        cls_input_mean, cls_input_var, cls_input0 = self.encode_input(X, training=training)
        return cls_input0,cls_input_mean, cls_input_var

    def spn_clf(self,embedding_org, training=False):
        y_pred = self.classifier(embedding_org)
        return y_pred,None


    def execute_net(self,X,training=True):
        normalized_X=tf.cast(X,tf.float32)#X/255.0
        if self.load_pretrain_model:
            normalized_X = tf.expand_dims(normalized_X[:,:,:,0],axis=-1)
            normalized_X/=255.0

        cls_input_mean,cls_input_var,cls_input0,decoding=self.enc_dec(X,training=training)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=decoding, labels=normalized_X)
        cros_ent_mean=tf.reduce_mean(cross_ent,axis=(1,2,3))
        kl_loss = -0.5 * tf.reduce_sum(1 + cls_input_var - tf.square(cls_input_mean) - tf.exp(cls_input_var), axis=1)
        loss = cros_ent_mean +(kl_loss*0.0001)
        l1_loss = tf.reduce_mean(cros_ent_mean)
        kl_loss = tf.reduce_mean(kl_loss)
        loss = tf.reduce_mean(loss)
        sigmoid_vals=tf.keras.activations.sigmoid(decoding)
        other_loss=[loss_fkt(sigmoid_vals,normalized_X) for loss_fkt in self.loss_fkts]

        #logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        #logpz = log_normal_pdf(cls_input0, 0., 0.)
        #logqz_x = log_normal_pdf(cls_input0, cls_input_mean, cls_input_var)
        #return -tf.reduce_mean(logpx_z + logpz - logqz_x),-tf.reduce_mean(logpx_z),-tf.reduce_mean(logpz - logqz_x)
        return loss,l1_loss,kl_loss,other_loss

    def model_execution_X(self,X, add_info_data, training=False):

        normalized_X=tf.cast(X,tf.float32)#X/255.0
        if self.load_pretrain_model:
            normalized_X=tf.expand_dims(normalized_X[:,:,:,0],axis=-1)
            normalized_X/=255.0

        cls_input_mean, cls_input_var, cls_input0 = self.encode_input(X, training=training)

        if self.add_info:
            cls_input=tf.concat([cls_input0,tf.cast(add_info_data,dtype=tf.float32)],axis=-1)
        else:
            cls_input=cls_input0

        y_pred=self.classifier(cls_input)
        return y_pred

    def reconstruct(self,z):
        decoding = self.decoder(z)
        sigmoid_vals = decoding#tf.keras.activations.sigmoid(decoding)
        return sigmoid_vals


    def execute_net_xy(self,X,y,add_info,training=True,beta=0):
        normalized_X=tf.cast(X,tf.float32)#X/255.0
        if self.load_pretrain_model:
            normalized_X=tf.expand_dims(normalized_X[:,:,:,0],axis=-1)
            normalized_X/=255.0

        cls_input_mean, cls_input_var, cls_input0 = self.encode_input(X, training=training)
        if self.VAE_fine_tune == 0:
            one_hot_y=tf.one_hot(y,2,dtype=dtype)

            decoder_input=tf.concat([cls_input0,one_hot_y],axis=-1)
        else:
            decoder_input=cls_input0

        if self.add_info:
            cls_input=tf.concat([cls_input0,tf.cast(add_info,dtype=tf.float32)],axis=-1)
        else:
            cls_input=cls_input0
        y_pred=self.classifier(cls_input)
        decoding = self.decoder(decoder_input)

        clf_loss=self.clf_loss(y,y_pred)
        sigmoid_vals =decoding #tf.keras.activations.sigmoid(decoding)


        rec_loss=tf.reduce_mean(tf.keras.losses.MSE(sigmoid_vals,normalized_X),axis=(1,2))
        MSE=tf.reduce_mean(tf.keras.losses.MSE(sigmoid_vals,normalized_X),axis=(1,2))
        l1_loss = tf.reduce_mean(tf.keras.losses.MAE(sigmoid_vals, normalized_X), axis=(1, 2))

        kl_loss = -0.5 * tf.reduce_sum(1 + cls_input_var - tf.square(cls_input_mean) - tf.exp(cls_input_var), axis=1)

        if self.use_KLD_anneal:
            loss = (rec_loss * self.loss_weights[0]) + (kl_loss * beta) + (clf_loss *self.loss_weights[2])
        else:
            loss = (rec_loss * self.loss_weights[0]) + (kl_loss * self.loss_weights[1]) + (clf_loss *self.loss_weights[2])
        xentro = tf.reduce_mean(rec_loss)
        kl_loss = tf.reduce_mean(kl_loss)
        loss = tf.reduce_mean(loss)
        clf_loss=tf.reduce_mean(clf_loss)
        other_loss=[tf.reduce_mean(l1_loss),tf.reduce_mean(MSE)]

        return loss,xentro,kl_loss,clf_loss,other_loss,y_pred,cls_input0,sigmoid_vals
    def evaluate_(self,dataset,verbose=0):
        losses = np.zeros((4 + 2))
        num_rounds = 0
        predictions = []
        gt = []
        if self.add_info:
            for test_x, y,other_info in dataset:
                loss, rec_loss, kld, clf, other_losses, pred,_,_ = self.execute_net_xy(test_x, y,other_info,training=False)
                losses[0] += loss
                losses[1] += rec_loss
                losses[2] += kld
                losses[3] += clf
                predictions.extend(pred.numpy().tolist())
                gt.extend(y.numpy().tolist())
                for i, tmp_loss in enumerate(other_losses):
                    losses[4 + i] += tf.reduce_mean(tmp_loss)

                num_rounds += 1
        else:
            for test_x, y in dataset:
                loss, rec_loss, kld, clf, other_losses, pred,_,_ = self.execute_net_xy(test_x, y,None, training=False)
                losses[0] += loss
                losses[1] += rec_loss
                losses[2] += kld
                losses[3] += clf
                predictions.extend(pred.numpy().tolist())
                gt.extend(y.numpy().tolist())
                for i, tmp_loss in enumerate(other_losses):
                    losses[4 + i] += tf.reduce_mean(tmp_loss)

                num_rounds += 1
        # calculate accuracy:
        softmax_pred = tf.nn.softmax(predictions, axis=-1)
        prediction_exponential = tf.math.exp(predictions)
        arg_max = np.argmax(softmax_pred, axis=-1)
        acc_loss = accuracy_score(gt, arg_max, normalize=True)
        losses = losses / num_rounds
        val_loss = losses[0]
        balanced_acc=balanced_accuracy_score(gt, arg_max)
        [prec,rec,f1,_]=precision_recall_fscore_support(gt, arg_max, average=None)

        pred_exp = np.nan_to_num(prediction_exponential[:, 1], nan=0, posinf=1.0)
        fpr, tpr, thresholds = metrics.roc_curve(gt, pred_exp,pos_label=1)

        auc=metrics.auc(fpr, tpr)

        return [acc_loss,val_loss,balanced_acc,prec[1],rec[1],f1[1],auc]

    def clf_model(self):
        inputs = tf.keras.Input(shape=self.input_shape_, dtype=dtype)
        encoding= self.encoder(inputs)
        cls_input_mean = self.z_mean(encoding)
        cls_input_var = self.z_log_var(encoding)
        clf_out=tf.keras.layers.Lambda(sampling, output_shape=(self.latent_dim,), name='z')([cls_input_mean, cls_input_var])


        cnn_embedding = tf.keras.Model(inputs=inputs, outputs=[clf_out,cls_input_mean,cls_input_var])
        return cnn_embedding

    def grad_cam_model(self):
    #    # Get the target layer
        target_layer = self.encoder.get_layer('conv2d')

        input1 = target_layer.input
        input2 = tf.keras.Input(shape=(3,), dtype=dtype)


        conv= target_layer(input1)

        encoding=None



        cls_input_mean = self.z_mean(encoding)
        cls_input_var = self.z_log_var(encoding)
        clf_out = tf.keras.layers.Lambda(sampling, output_shape=(self.latent_dim,), name='z')([cls_input_mean, cls_input_var])
        cls_input=tf.concat([clf_out,tf.cast(input2,dtype=tf.float32)],axis=-1)
        output = self.classifier(cls_input)

        # Create a model that maps the input image to the activations of the target layer
        grad_model = tf.keras.models.Model(
            [input1,input2], [target_layer.output,output]
        )
        return grad_model





def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon










