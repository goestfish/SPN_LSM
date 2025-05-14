import time
import random
import sys
print(sys.modules.get('tensorflow'))
import tensorflow as tf #.compat.v2
print(tf.__version__)
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support



def create_embedding_model(input_dimensions):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=input_dimensions))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='relu'))
    return model



class CNN_SPN(tf.keras.Model):
    def __init__(self,num_classes,input_dimensions,learning_rate,spn=None,cnn=None,get_max=True):
        super(CNN_SPN, self).__init__()
        #self.spn=spn
        self.num_classes=num_classes
        self.get_max=get_max
        # 1. define CNN / MLP
        if cnn!=None:
            self.embedding=cnn
        else:
            self.embedding=create_embedding_model(input_dimensions)
        # 2. add SPN if it is given
        if spn is not None:
            self.spn_training=True
            self.clf =spn
        # 3. else create a classifier

        else:
            self.clf=Dense(1, activation='sigmoid')
            self.clf_loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        self.optimizer=tf.keras.optimizers.Adam(learning_rate)
        #self.optimizer = tf.keras.optimizers.SGD(learning_rate)

    def model_execution_X_y(self,X,y,other):
        embedding=self.embedding(X)
        #em=embedding.numpy
        if self.spn_training:
            # get max values or whatever:
            if self.get_max:
                embedding = tf.math.reduce_max(embedding, axis=(1, 2))
            if self.use_add_info:
                embedding=tf.concat([embedding,other],axis=-1)
            spn_input=tf.concat([tf.cast(tf.expand_dims(y,axis=-1),dtype=tf.float32),embedding],axis=-1)
            #spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
            spn_output=self.clf(spn_input)
            #loss = -tf.reduce_sum(tf_graph)
            #opt_op = optimizer.minimize(loss)
            loss=tf.reduce_sum(spn_output)
        else:
            # xEntropy
            loss=self.clf_loss(y,embedding)
        return loss

    @tf.function
    def train_step(self,x,y,other):
        with tf.GradientTape() as tape:
            loss=self.model_execution_X_y(x,y,other)


        #test=self.clf.trainable_variables
        #print(self.clf.trainable_variables)
        train_vars = self.trainable_variables#+test
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return loss

    #@tf.function
    def train(self,train_ds,first_loss):
        all_losses=0.0
        counter=0.0
        for it_c,train_rec in enumerate(train_ds):
            other=0
            if self.use_add_info:
                (X, y,other) = train_rec
                other=tf.cast(other,tf.float32)
            else:
                (X, y)=train_rec

            loss=self.train_step(X,y,other)
            all_losses+=loss.numpy()
            counter+=1
            if not it_c and first_loss:
                print(it_c,'loss',all_losses/counter)
        return all_losses/counter

    def eval_cnn(self,test_X):
        return self.embedding(test_X)
    def get_spn_variables(self):
        pass
def enc_layer(net,stride,filter_num,batchnorm_integration,shortcut,
              dilations,filter_size,trainable,dtype,old_h, old_num_filter,dropout,activation,
              name):

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

    convolution = activation(convolution)
    return convolution
def make_costume_encoder(filter_size, num_layer, input_shape, batchnorm_integration, num_filter, shortcut, strides,
                         activation, encoder_name, dtype, dilations, dropout, end_dim):
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
                        name=encoder_name + '_' + str(layer_num))
        img_size /= stride

    # net = tf.keras.layers.MaxPooling2D(pool_size=(img_size, img_size))(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(end_dim, activation=None)(net)
    net = activation(net)

    net=tf.keras.layers.Dense(1)(net)



    model = tf.keras.Model(inputs=inputs, outputs=net)
    return model


def gradient_penalty(critic, real_images, fake_images):
    batch_size = tf.shape(real_images)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        interpolated_output = critic(interpolated)
    grads = tape.gradient(interpolated_output, interpolated)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3])+ 1e-8)
    penalty = tf.reduce_mean((grad_norm - 1.0) ** 2)
    return penalty

class CNN_SPN_Parts(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 learning_rate,
                 all_spn_x_y_model,
                 all_prior,
                 cnn=None,
                 get_max=True,
                 dtype=tf.dtypes.float32,
                 gauss_embeds=0.01,
                 use_add_info=False,
                 VAE_fine_tune=0,
                 decoder=None,
                 loss_weights=[],
                 load_pretrain_model=1,
                 use_GAN=False,

                 filter_size=3,
                 num_layer=4,
                 input_shape=(128,128,1),
                 batchnorm_integration=1,
                 shortcut=0,
                 activation='relu',
                 num_filter=[64,128,128],
                 strides=[2,2,2,2],
                 end_dim=100,
                 dropout=0.2,
                 dilations=1,
                 discriminator_name='gan_discriminator',
                 clf_mlp=None

                 ):
        super(CNN_SPN_Parts, self).__init__()
        self.clf_mlp=clf_mlp
        self.load_pretrain_model=load_pretrain_model
        self.all_spn_x_y=all_spn_x_y_model
        self.use_add_info=use_add_info
        self.decoder=decoder
        self.VAE_fine_tune=VAE_fine_tune
        self.loss_weights=loss_weights
        #self.dtype=dtype
        #self.parallel_iterations=len(all_prior)
        #TODO use softmax inverse?
        self.prior_weights = tf.Variable(initial_value=all_prior, dtype=dtype, trainable=False, name='Root weights')  # name="mean",

        #self.spn=spn
        self.num_classes=num_classes
        self.get_max=get_max
        # 1. define CNN / MLP
        self.embedding=cnn

        # 2. add SPN if it is given
        #if spn is not None:
        self.spn_training=True
        self.clf =all_spn_x_y_model
        # 3. else create a classifier

        #else:
        #    self.clf=Dense(1, activation='sigmoid')
        self.clf_loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.gauss_embed=gauss_embeds
        if gauss_embeds:
            self.gauss_layer=tf.keras.layers.GaussianNoise(gauss_embeds)

        self.optimizer=tf.keras.optimizers.Adam(learning_rate)
        #self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)

        self.use_GAN=use_GAN
        if self.use_GAN:
            if activation == 'relu':
                self.activation = tf.keras.activations.relu
            elif activation == 'tanh':
                self.activation = tf.keras.activations.tanh
            # create discriminator:
            #self.discriminator=make_pretrain_discriminator(3, 1, (128,128,3), 1, 64,'relu', 0,path='')
            self.discriminator=make_costume_encoder(filter_size, num_layer, input_shape, batchnorm_integration, num_filter, shortcut, strides,
                         self.activation, discriminator_name, dtype, dilations, dropout, end_dim)

            # create additional loss:
            #self.gan_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            # gan opt:
            self.gan_discr_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
            self.gan_gener_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)



    def model_execution_X_y(self,X,y):
        [embedding,_,_]=self.embedding(X)

        if self.get_max:
            embedding = tf.math.reduce_max(embedding, axis=(1, 2))
        if self.gauss_embed:
            embedding=self.gauss_layer(embedding)
        spn_input=tf.concat([tf.cast(tf.expand_dims(y,axis=-1),dtype=tf.float32),embedding],axis=-1)


        tf_weights = tf.expand_dims(tf.expand_dims(self.prior_weights, axis=0), axis=-1)
        # Stack children along axis 1
        inputs = []
        for sub_spn in self.all_spn_x_y:
            inputs.append(sub_spn(spn_input))

        children_prob = tf.stack(inputs, axis=1)  # children
        # print('childrenprob shape ',children_prob.shape)
        # Calculate log-sum-exp
        log_enumerator = children_prob + tf.math.log(tf_weights)
        p_x = tf.reduce_logsumexp(log_enumerator, axis=1)
        p_y_x = log_enumerator - p_x

        return p_y_x


    def spn_clf(self,embedding,training):
        inputs = []
        tf_weights = tf.expand_dims(self.prior_weights, axis=0)
        for label_id, sub_spn in enumerate(self.all_spn_x_y):
            y=np.full((embedding.shape[0],1),label_id)
            spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
            inputs.append(sub_spn(spn_input,training=training))

        children_prob = tf.concat(inputs, axis=1)  # children
        # print('childrenprob shape ',children_prob.shape)
        # Calculate log-sum-exp
        log_enumerator = children_prob + tf.math.log(tf_weights)
        p_x = tf.reduce_logsumexp(log_enumerator, axis=1,keepdims=True)
        p_y_x = log_enumerator - p_x
        return p_y_x,p_x
    #'''
    def model_execution_X(self, X,other_data,training=True):
        [embedding,_,_] = self.embedding(X,training=training)
        # em=embedding.numpy
        # if self.spn_training:
        # get max values or whatever:
        if self.get_max:
            embedding = tf.math.reduce_max(embedding, axis=(1, 2))

        # spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
        if self.gauss_embed and training:
            embedding=self.gauss_layer(embedding)

        # Stack children along axis 1
        if self.use_add_info:
            embedding=tf.concat([embedding,other_data],axis=-1)

        p_y_x,p_x=self.spn_clf(embedding, training)


        # spn_output=self.clf(spn_input)
        return p_y_x


    @tf.function
    def train_step(self,x,y,other_data):
        with tf.GradientTape() as tape:
            spn_out=self.model_execution_X(x,other_data)
            loss = self.clf_loss(y, spn_out)


        #test=self.clf.trainable_variables
        #print(self.clf.trainable_variables)
        train_vars = self.trainable_variables#+test
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return [loss]

    def reconstruct(self,embedding):
        reconstruction = self.decoder(embedding)
        #sigmoid_vals = tf.keras.activations.sigmoid(reconstruction)
        return reconstruction

    def vae_rec(self,X):
        [embedding, _, _] = self.embedding(X, training=False)
        return self.reconstruct(embedding)



    def model_execution_vae(self, X,y_real,other_data,training=True):
        normalized_X = tf.cast(X, tf.float32)
        if self.load_pretrain_model:
            normalized_X=tf.expand_dims(normalized_X[:,:,:,0],axis=-1)
            normalized_X/=255.0
        [embedding_,embed_mean,embed_var] = self.embedding(X,training=training)
        #one_hot_y=tf.one_hot(y,2,dtype=dtype)
        #decoder_input=tf.concat([cls_input0,one_hot_y],axis=-1)
        embedding=tf.identity(embedding_)
        reconstruction=self.decoder(embedding)
        kl_loss = -0.5 * tf.reduce_sum(1 + embed_var - tf.square(embed_mean) - tf.exp(embed_var), axis=1)

        if self.get_max:
            embedding = tf.math.reduce_max(embedding_, axis=(1, 2))

        # spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
        if self.gauss_embed and training:
            embedding=self.gauss_layer(embedding)
        tf_weights = tf.expand_dims(self.prior_weights, axis=0)
        # Stack children along axis 1
        inputs = []
        if self.use_add_info:
            embedding=tf.concat([embedding,other_data],axis=-1)
        for label_id, sub_spn in enumerate(self.all_spn_x_y):
            y=np.full((embedding.shape[0],1),label_id)
            spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
            inputs.append(sub_spn(spn_input,training=training))

        children_prob = tf.concat(inputs, axis=1)  # children
        # print('childrenprob shape ',children_prob.shape)
        # Calculate log-sum-exp
        log_enumerator = children_prob + tf.math.log(tf_weights)
        p_x = tf.reduce_logsumexp(log_enumerator, axis=1,keepdims=True)
        p_y_x = log_enumerator - p_x
        clf_loss=self.clf_loss(y_real, p_y_x)
        sigmoid_vals =reconstruction #tf.keras.activations.sigmoid(reconstruction)
        rec_loss=tf.reduce_mean(tf.keras.losses.MSE(sigmoid_vals,normalized_X),axis=(1,2))


        loss = (rec_loss * 2*self.loss_weights[0]) + (kl_loss * self.loss_weights[1]) + (clf_loss * self.loss_weights[2])

        rec_loss=tf.reduce_mean(rec_loss)
        kl_loss=tf.reduce_mean(kl_loss)
        loss=tf.reduce_mean(loss)

        return p_y_x,loss,rec_loss,clf_loss,kl_loss,embedding_

    def model_execution_vae_eval(self, X,y_real,other_data):
        training=False
        normalized_X = tf.cast(X, tf.float32)
        if self.load_pretrain_model:
            normalized_X=tf.expand_dims(normalized_X[:,:,:,0],axis=-1)
            normalized_X/=255.0
        [embedding_,embed_mean,embed_var] = self.embedding(X,training=training)
        #one_hot_y=tf.one_hot(y,2,dtype=dtype)
        #decoder_input=tf.concat([cls_input0,one_hot_y],axis=-1)
        embedding=tf.identity(embedding_)
        reconstruction=self.decoder(embedding)
        kl_loss = -0.5 * tf.reduce_sum(1 + embed_var - tf.square(embed_mean) - tf.exp(embed_var), axis=1)

        if self.get_max:
            embedding = tf.math.reduce_max(embedding_, axis=(1, 2))

        # spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
        if self.gauss_embed and training:
            embedding=self.gauss_layer(embedding)
        tf_weights = tf.expand_dims(self.prior_weights, axis=0)
        # Stack children along axis 1
        inputs = []
        if self.use_add_info:
            embedding=tf.concat([embedding,other_data],axis=-1)
        for label_id, sub_spn in enumerate(self.all_spn_x_y):
            y=np.full((embedding.shape[0],1),label_id)
            spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
            inputs.append(sub_spn(spn_input,training=training))
        p_y_x_mlp=self.clf_mlp(embedding)

        children_prob = tf.concat(inputs, axis=1)  # children
        # print('childrenprob shape ',children_prob.shape)
        # Calculate log-sum-exp
        log_enumerator = children_prob + tf.math.log(tf_weights)
        p_x = tf.reduce_logsumexp(log_enumerator, axis=1,keepdims=True)
        p_y_x = log_enumerator - p_x
        clf_loss=self.clf_loss(y_real, p_y_x)
        sigmoid_vals =reconstruction #tf.keras.activations.sigmoid(reconstruction)
        rec_loss=tf.reduce_mean(tf.keras.losses.MSE(sigmoid_vals,normalized_X),axis=(1,2))
        mae = tf.reduce_mean(tf.keras.losses.MAE(sigmoid_vals, normalized_X), axis=(1, 2))


        loss = (rec_loss * 2*self.loss_weights[0]) + (kl_loss * self.loss_weights[1]) + (clf_loss * self.loss_weights[2])

        rec_loss=tf.reduce_mean(rec_loss)
        kl_loss=tf.reduce_mean(kl_loss)
        loss=tf.reduce_mean(loss)
        mae=tf.reduce_mean(mae)

        return p_y_x,loss,rec_loss,clf_loss,kl_loss,mae,embedding_,p_y_x_mlp

    @tf.function
    def train_step_vae_one_loss(self,x,y,other_data):
        with tf.GradientTape() as tape:
            spn_out,loss,rec_loss,clf_loss,kl_loss,z=self.model_execution_vae(x,y,other_data)
        train_vars=self.embedding.trainable_variables+self.decoder.trainable_variables
        for entry in self.all_spn_x_y:
            train_vars.extend(entry.trainable_variables)

        #train_vars = #+[entry.trainable_variables ]
        #print('VAE train var names')
        #for var in train_vars:
        #    if hasattr(var, 'name'):
        #        print(var.name, end=';')
        #    else:
        #        print()
        #        print('object type', type(var),)
        #self.trainable_variables#+test
        # TODO check if VAE variables are also inside (decoder)
        #print(' TODO check if VAE variables are also inside (decoder)')
        #print(train_vars)
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return [loss,rec_loss,clf_loss,kl_loss],z,spn_out

    @tf.function
    def gan_step(self,X, y, other_info,discriminator_loss_old, generator_loss_old):# GAN Step (Discriminator training)

        #if discriminator_loss_old-generator_loss_old<0.2:
        for _ in range(5):
            with tf.GradientTape() as tape:
                fake_images = self.decoder(self.embedding(X)[0])
                # copy fake images:
                #fake_images=tf.expand_dims(fake_images,axis=-1)
                #fake_images=tf.tile(fake_images, multiples=[1,1, 1, 3])
                real_output = self.discriminator(X[:,:,:,0:1]/255)
                fake_output = self.discriminator(tf.keras.activations.sigmoid(fake_images))  # Compute once
                #real_loss = self.gan_loss_fn(tf.ones_like(real_output), real_output)
                #fake_loss = self.gan_loss_fn(tf.zeros_like(fake_output), fake_output)
                gp = gradient_penalty(self.discriminator, X[:,:,:,0:1]/255, tf.keras.activations.sigmoid(fake_images))

                gan_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)+ 10*gp  # Wasserstein loss

                #gan_loss = real_loss + fake_loss
            grads = tape.gradient(gan_loss, self.discriminator.trainable_weights)
            #clipped_grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]
            self.gan_discr_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
            #else:
            #    gan_loss=discriminator_loss_old

        with tf.GradientTape() as tape:
            # Use the same `fake_output` computed earlier
            fake_images = self.decoder(self.embedding(X)[0])
            #fake_images = tf.tile(fake_images, multiples=[1, 1, 1, 3])
            fake_output = self.discriminator(tf.keras.activations.sigmoid(fake_images))  # Compute once
            #generator_loss = self.gan_loss_fn(tf.ones_like(fake_output), fake_output)
            generator_loss = -tf.reduce_mean(fake_output)  # Wasserstein loss
        grads = tape.gradient(generator_loss, self.decoder.trainable_weights)
        #clipped_grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]
        self.gan_gener_optimizer.apply_gradients(zip(grads, self.decoder.trainable_weights))

        return gan_loss,generator_loss
    def train_gan(self,train_ds,first_loss):
        counter=0.0

        #train_fkt=self.train_step_vae_one_loss
        all_losses = np.zeros(6)
        z_train, gt_train, predictions_train=[],[],[]
        discriminator_loss_old, generator_loss_old=0,0
        for it_c, train_rec in enumerate(train_ds):
            other_info = 0
            if self.use_add_info:
                (X, y, other_info) = train_rec
                other_info = tf.cast(other_info, tf.float32)
            else:
                (X, y) = train_rec

            # VAE step:
            loss, z, pred = self.train_step_vae_one_loss(X, y, other_info)
            # GAN step:
            discriminator_loss,generator_loss=self.gan_step(X,y,other_info,discriminator_loss_old, generator_loss_old)

            z_train.append(z.numpy())
            gt_train.append(y)
            predictions_train.append(pred.numpy())


            new_loss = [entry.numpy() for entry in loss]
            new_loss.append(discriminator_loss.numpy())
            new_loss.append(generator_loss.numpy())
            discriminator_loss_old, generator_loss_old=discriminator_loss.numpy(),generator_loss.numpy()
            #print(new_loss)
            #for e in new_loss:
            #    print(e.shape)
            all_losses += new_loss
            counter += 1
            if not it_c and first_loss:
                print(it_c, 'loss', all_losses / counter)
        return [all_losses / counter,z_train, gt_train,predictions_train]



    @tf.function
    def train_step_vae_diff_loss(self,x,y,other_data):
        with tf.GradientTape() as tape:
            spn_out=self.model_execution_X(x,other_data)
            loss = self.clf_loss(y, spn_out)

        train_vars = self.trainable_variables#+test
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return loss




    #@tf.function
    def train(self,train_ds,first_loss):
        counter=0.0
        if self.VAE_fine_tune==0:
            #train_fkt=self.train_step
            all_losses = np.zeros(1)
            for it_c, train_rec in enumerate(train_ds):
                other_info = 0
                if self.use_add_info:
                    (X, y, other_info) = train_rec
                    other_info = tf.cast(other_info, tf.float32)
                else:
                    (X, y) = train_rec

                loss = self.train_step(X, y, other_info)
                new_loss = [entry.numpy() for entry in loss]
                all_losses += new_loss
                counter += 1
                if not it_c and first_loss:
                    print(it_c, 'loss', all_losses / counter)
            return all_losses / counter
        elif self.VAE_fine_tune==1:
            #train_fkt=self.train_step_vae_one_loss
            all_losses = np.zeros(4)
            z_train, gt_train, predictions_train=[],[],[]
            for it_c, train_rec in enumerate(train_ds):
                other_info = 0
                if self.use_add_info:
                    (X, y, other_info) = train_rec
                    other_info = tf.cast(other_info, tf.float32)
                else:
                    (X, y) = train_rec

                loss,z,pred = self.train_step_vae_one_loss(X, y, other_info)

                z_train.append(z.numpy())
                gt_train.append(y)
                predictions_train.append(pred.numpy())


                new_loss = [entry.numpy() for entry in loss]

                all_losses += new_loss
                counter += 1
                if not it_c and first_loss:
                    print(it_c, 'loss', all_losses / counter)
            return [all_losses / counter,z_train, gt_train,predictions_train]

        elif self.VAE_fine_tune==2:

            all_losses = np.zeros(4)


            for it_c, train_rec in enumerate(train_ds):
                other_info=0
                if self.use_add_info:
                    (X,y,other_info)=train_rec
                    other_info=tf.cast(other_info,tf.float32)
                else:
                    (X, y)=train_rec

                loss=self.train_step_vae_diff_loss(X,y,other_info)
                new_loss=[entry.numpy() for entry in loss]
                all_losses+=new_loss
                counter+=1
                if not it_c and first_loss:
                    print(it_c,'loss',all_losses/counter)
            return all_losses/counter






    def eval_cnn(self,test_X):
        return self.embedding(test_X)
    def get_spn_variables(self):
        pass




def train_model_parts(grid_params,cnn_spn, train_data, val_data,test_data, num_iterations,ckpt,manager,val_entropy,val_acc=0,add_info=False):
    first_loss=True

    print('number of trainable variables in cnn spn:',len(cnn_spn.trainable_variables))

    #best_val_loss = val_entropy
    train_start_time = time.time()
    best_acc_loss=val_acc
    best_val_reconstruction=100000
    eval_after_train = []
    all_debugging_stuff = []
    for i in range(num_iterations):

        idx_train = list(range(train_data[0].shape[0]))
        random.shuffle(idx_train)

        X = train_data[0][idx_train]
        if add_info:
            y = train_data[1][idx_train,0]
            other=train_data[1][idx_train,1:]
            if grid_params.use_add_info:
                train_dataset = tf.data.Dataset.from_tensor_slices((X, y, other)).batch(grid_params.batch_size)
            else:
                train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(grid_params.batch_size)


        else:
            y = train_data[1][idx_train]
            train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(grid_params.batch_size)
        if grid_params.VAE_fine_tune and not grid_params.GAN:
            [train_loss,z_train, gt_train,predictions_train]=cnn_spn.train(train_dataset,first_loss=first_loss)
        elif grid_params.VAE_fine_tune and grid_params.GAN:
            [train_loss, z_train, gt_train, predictions_train] = cnn_spn.train_gan(train_dataset, first_loss=first_loss)
        else:
            train_loss= cnn_spn.train(train_dataset, first_loss=first_loss)
        first_loss=False

        if i%3==0:
            if grid_params.VAE_fine_tune:
                val_losses,z,gt,predictions ,_= test_model_all(cnn_spn, val_data, num_classes=2, batch_size=grid_params.batch_size,
                                                 training=False, add_info=add_info)
                train_rec=cnn_spn.vae_rec(train_data[0][:16])
                test_rec=cnn_spn.vae_rec(test_data[0][:16])

                all_debugging_stuff.append([[val_losses, z, gt, test_rec, predictions],
                                            [train_loss, z_train, gt_train, train_rec, predictions_train]])
            else:
                [val_losses]=test_model_no_mpe(cnn_spn, [val_data], num_classes=2, batch_size=grid_params.batch_size, training=False,add_info=add_info)


            improve = False
            #if val_loss < best_val_loss:
            if val_losses[2] > best_acc_loss or (val_losses[2] > (best_acc_loss-0.01) and val_losses[-1][1]<best_val_reconstruction):
                best_val_reconstruction=val_losses[-1][1]
                best_acc_loss=val_losses[2]
                #best_val_loss = val_loss
                # Save the model weights
                ckpt[0].step.assign_add(1)
                manager[0].save()

                ckpt[1].step.assign_add(1)
                manager[1].save()

                improve=True
                eval_after_train = test_model_no_mpe(cnn_spn, [train_data, val_data, test_data], cnn_spn.num_classes,
                                                     grid_params.batch_size,add_info=add_info)

                # TODO save VAE:

            print('val entropy', i, val_losses[1],'improve',improve,'curr acc', val_losses[0], val_losses[2],'best acc',best_acc_loss)
            print('curr rec',val_losses[-1][1],'best rec:',best_val_reconstruction)
        print('loss', i,end=':' )
        for loss in train_loss:
            print(np.round(loss,5),end=' - ')
        print()

    print('fine tune train time',(time.time()-train_start_time)//60,)
    return eval_after_train,all_debugging_stuff




def test_model_no_mpe(tfmodel, data_sets,num_classes,batch_size=25,training=False,add_info=False):
    all_evals=[]
    for dataset in data_sets:
        prediction=[]
        gt=[]
        for i in range(dataset[0].shape[0]//batch_size):
            X_data = dataset[0][i * batch_size:(i + 1) * batch_size]
            other_data=0
            if add_info:

                y_data=dataset[1][i*batch_size:(i+1)*batch_size,0]
                other_data=dataset[1][i*batch_size:(i+1)*batch_size,1:]

            else:

                y_data=dataset[1][i*batch_size:(i+1)*batch_size]

            pred=tfmodel.model_execution_X(X_data,other_data,training=False)
            prediction.extend(pred.numpy().tolist())
            gt.extend(y_data.tolist())


        prediction=np.asarray(prediction)
        entropy=tfmodel.clf_loss(gt,prediction).numpy()
        prediction_exponential=tf.math.exp(prediction)

        arg_max=np.argmax(prediction_exponential,axis=-1)
        acc = accuracy_score(gt, arg_max, normalize=True)

        auc=0
        if num_classes==2:
            test1 = np.isnan(prediction_exponential[:,1])

            pred_exp=np.nan_to_num(prediction_exponential[:,1],nan=0,posinf=1.0)
            fpr, tpr, thresholds = metrics.roc_curve(gt, pred_exp,pos_label=1)
            auc = metrics.auc(fpr, tpr)
        balanced_acc=balanced_accuracy_score(gt, arg_max)
        [prec,rec,f1,_]=precision_recall_fscore_support(gt, arg_max, average=None)
        results=[acc,entropy,balanced_acc,prec[1],rec[1],f1[1],auc]
        all_evals.append(results)
        print(results)
    #print(all_evals)
    return all_evals


def test_model_SPN_MLP(tfmodel, dataset,num_classes,batch_size=25,training=False,add_info=False):
    mlp_prediction=[]
    prediction=[]
    gt=[]
    z=[]
    losses=np.zeros(5)
    for i in range(dataset[0].shape[0]//batch_size):
        X_data = dataset[0][i * batch_size:(i + 1) * batch_size]
        other_data=0
        if add_info:

            y_data=dataset[1][i*batch_size:(i+1)*batch_size,0]
            other_data=dataset[1][i*batch_size:(i+1)*batch_size,1:]

        else:

            y_data=dataset[1][i*batch_size:(i+1)*batch_size]


        pred, loss, rec_loss, clf_loss, kl_loss, mae,embedding_,mlp_pred=tfmodel.model_execution_vae_eval( X_data, y_data, other_data)
        new_loss=np.asarray([loss.numpy(), rec_loss.numpy(), clf_loss.numpy(), kl_loss.numpy(),mae.numpy()])
        losses+=new_loss
        prediction.extend(pred.numpy().tolist())
        mlp_prediction.extend(mlp_pred.numpy().tolist())
        gt.extend(y_data.tolist())
        z.extend(embedding_.numpy().tolist())
    losses/=dataset[0].shape[0]//batch_size
    prediction=np.asarray(prediction)
    mlp_prediction=np.asarray(mlp_prediction)


    prediction_exponential=tf.math.exp(prediction)

    results_MLP,_,_ =eval_cls(mlp_prediction,mlp_prediction,gt,tfmodel.clf_loss, num_classes)
    results_SPN,_,_=eval_cls(prediction, prediction_exponential, gt, tfmodel.clf_loss, num_classes)



    return results_MLP,results_SPN,losses


def test_model_all(tfmodel, dataset, num_classes, batch_size=25, training=False, add_info=False):
    #mlp_prediction = []
    prediction = []
    gt = []
    z = []
    losses = np.zeros(4)
    for i in range(dataset[0].shape[0] // batch_size):
        X_data = dataset[0][i * batch_size:(i + 1) * batch_size]
        other_data = 0
        if add_info:

            y_data = dataset[1][i * batch_size:(i + 1) * batch_size, 0]
            other_data = dataset[1][i * batch_size:(i + 1) * batch_size, 1:]

        else:

            y_data = dataset[1][i * batch_size:(i + 1) * batch_size]

        pred, loss, rec_loss, clf_loss, kl_loss, mae,embedding_, mlp_pred = tfmodel.model_execution_vae_eval(X_data, y_data,
                                                                                                         other_data)
        new_loss = np.asarray([loss.numpy(), rec_loss.numpy(), clf_loss.numpy(), kl_loss.numpy()])
        losses += new_loss
        prediction.extend(pred.numpy().tolist())
        #mlp_prediction.extend(mlp_pred.tolist())
        gt.extend(y_data.tolist())
        z.extend(embedding_.numpy().tolist())
    losses /= dataset[0].shape[0] // batch_size
    prediction = np.asarray(prediction)

    prediction_exponential = tf.math.exp(prediction)

    results,pred_exp,arg_max = eval_cls(prediction, prediction_exponential, gt, tfmodel.clf_loss, num_classes)
    return results, z, gt, arg_max, pred_exp

def eval_cls(pred_logits,prediction_exponential,gt,clf_loss,num_classes=2):
    pred_arg_max = np.argmax(prediction_exponential, axis=-1)
    entropy=clf_loss(gt,pred_logits).numpy()
    acc = accuracy_score(gt, pred_arg_max, normalize=True)

    auc=0
    pred_exp=0
    if num_classes==2:
        pred_exp=np.nan_to_num(prediction_exponential[:,1],nan=0,posinf=1.0)
        fpr, tpr, thresholds = metrics.roc_curve(gt, pred_exp,pos_label=1)
        auc = metrics.auc(fpr, tpr)
    balanced_acc=balanced_accuracy_score(gt, pred_arg_max)
    [prec,rec,f1,_]=precision_recall_fscore_support(gt, pred_arg_max, average=None)

    return [acc,entropy,balanced_acc,prec[1],rec[1],f1[1],auc],pred_exp,pred_arg_max
