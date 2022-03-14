import numpy as np
from matplotlib import pyplot as plt

import tensorflow
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, Dropout, BatchNormalization, Reshape, Flatten, Dense, MaxPooling2D, LeakyReLU
from tensorflow.keras.optimizers import Adam

print("Imported successfully!")

class Gan:
    def __init__(self,
        lr = 0.0002,
        beta_1 = 0.9,
        beta_2 = 0.999,
        epsilon = 1e-7,
        amsgrad = False,
        latent_dim = 100,
        n_samples = 25,
        disc_n_iter = 100,
        disc_n_batch = 256,
        n_batch = 256,
        n_epochs = 100,
        disc_flag = True,
        gen_flag = True):

        # training datasets
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None

        # input shapes
        self.train_x_shape = None
        self.test_x_shape = None
        self.train_y_shape = None
        self.test_y_shape = None

        # hyperparameters Adam
        self.learning_rate = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

        # latent space
        self.latent_dim = latent_dim
        self.n_samples = n_samples

        # hyperparameters Discriminator Training
        self.disc_n_iter = disc_n_iter
        self.disc_n_batch = disc_n_batch

        # general hyperparameters
        self.n_batch = n_batch
        self.n_epochs = n_epochs

        # auxiliary variables
        self.discriminator_flag = disc_flag
        self.generator_flag = gen_flag

    def load_dataset(self):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = load_data()
        self.train_x_shape = self.train_x.shape
        self.test_x_shape = self.test_x.shape
        self.train_y_shape = self.train_y.shape
        self.test_y_shape = self.test_y.shape
    
    def show_data(self, x_data):
        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.axis('off')
            plt.imshow(x_data[i], cmap='gray_r')
        plt.show()
    
    def _getInputShape(self):

        # get the input shape pattern of the dataset
        if K.image_data_format() == 'channels_first':
            input_shape = (1, 28, 28)
        else:
            input_shape = (28, 28, 1)
        return input_shape
    
    def discriminator(self):
        # retrieve the input shape pattern
        input_shape = self._getInputShape()
        
        # define sequential model
        model = Sequential()

        # convolutional architecture
        model.add(Conv2D(32, (3,3), padding = 'same', input_shape=input_shape))
        model.add(LeakyReLU(0.2))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

        model.add(Conv2D(64, (3,3), padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(Conv2D(128, (3,3), padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3,3), padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Conv2D(512, (3,3), padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(0.2))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        if self.discriminator_flag:
            model.summary()

        # optimizer
        opt = Adam(learning_rate=self.learning_rate, 
                        beta_1=self.beta_1, 
                        beta_2=self.beta_2, 
                        epsilon=self.epsilon, 
                        amsgrad=self.amsgrad)

        # model compiler
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model
    
    def real_images_samples(self):

        # get the training dataset
        x = self.train_x

        # expand to 3d, e.g. add channels dimension
        x = np.expand_dims(x, axis = -1)

        # convert to float
        x = x.astype('float32')

        # normalizing 
        x = x / 255.

        return x
    
    def generate_real_samples(self, n_samples):
        dataset = self.real_images_samples()
        i = np.random.randint(0, dataset.shape[0], n_samples)
        x = dataset[i]
        y = np.ones((n_samples, 1))
        return x, y
    
    def generate_fake_samples(self, n_samples):
        x = np.random.rand(28 * 28 * n_samples)
        x = x.reshape((n_samples, 28, 28, 1))
        y = np.zeros((n_samples,1))
        return x, y

    def train_discriminator(self, d_model):
        half_batch = self.disc_n_batch // 2
        for i in range(self.disc_n_iter):
            # get randomly real examples
            x_real, y_real = self.generate_real_samples(half_batch)
            # update discriminator on real samples
            _, real_acc = d_model.train_on_batch(x_real, y_real)
            # generate fake examples
            x_fake, y_fake = self.generate_fake_samples(half_batch)
            # update discriminator on fake samples
            _, fake_acc = d_model.train_on_batch(x_fake, y_fake)
            # summarize performance
            print(f'{i+1}: real_acc = {real_acc*100}%; fake_acc = {fake_acc*100}%')
    
    def generator(self):
        model = Sequential()
        # foundation for 7x7 image
        n_nodes = 128 * 7 * 7
        model.add(Dense(n_nodes, input_dim=self.latent_dim))
        model.add(LeakyReLU(0.2))
        model.add(Reshape((7, 7, 128)))
        # upsample to 14x14
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(0.2))
        # upsample to 28x28
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))

        if self.generator_flag:
            model.summary()

        return model
    
    def generate_latent_points(self, n_samples):
        # generate points in the latent space
        x_input = np.random.randn(self.latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, self.latent_dim)

        return x_input

    def generate_fake_samples_generator(self, g_model, n_samples):
        ''' An alterative method to generate fake samples using the
        generator.
        '''
        # generate points in the latent space
        x_input = self.generate_latent_points(n_samples)
        # predict outputs
        x = g_model.predict(x_input)
        y = np.zeros((self.n_samples,1))

        return x, y


    def gan(self, d_model, g_model):
        # make the weights in the discriminator not trainable
        d_model.trainable = False
        # connect both models
        model = Sequential()
        # add generator
        model.add(g_model)
        # add discriminator
        model.add(d_model)
        # define optimizer
        opt = Adam(learning_rate=self.learning_rate, 
                        beta_1=self.beta_1, 
                        beta_2=self.beta_2, 
                        epsilon=self.epsilon, 
                        amsgrad=self.amsgrad)
        # compile model
        model.compile(loss='binary_crossentropy', optimizer=opt)
        model.summary()
        return model
    
    def train(self, d_model, g_model, gan_model):
        # load dataset
        self.load_dataset()
        # define batches
        batch_per_epoch = self.train_x_shape[0] // self.n_batch
        half_batch = self.n_batch // 2
        #manually enumerates epochs
        for i in range(self.n_epochs):
            # enumerate batches over the training set
            for j in range(batch_per_epoch):
                # get randomly selected real samples
                x_real, y_real = self.generate_fake_samples(half_batch)
                # generate fake samples
                x_fake, y_fake = self.generate_fake_samples_generator(g_model, half_batch)
                # creating training stack for the discriminator
                x, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
                # update the discriminator model weights
                d_loss,_ = d_model.train_on_batch(x, y)
                # prepare points in latent space as input for the generator
                x_gan = self.generate_latent_points(self.n_batch)
                # create inverted labels for the fake samples
                y_gan = np.ones((self.n_batch,1))
                # update the generator via the discriminator's error
                g_loss,_ = gan_model.train_on_batch(x_gan, y_gan)
                # summarize loss of this batch
                print(f'>{i+1}, {j+1}/{batch_per_epoch}, d_loss={d_loss}, g_loss={g_loss}')
    
    def summarize_performance(self, d_model, g_model, n_samples = 100):
        # prepare real samples
        x_real, y_real = self.generate_real_samples(n_samples)
        # evaluate discriminator on real examples
        _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake = self.generate_fake_samples_generator(g_model, n_samples)
        # evaluate discriminator on fake examples
        _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)     
        # summarize discriminator performance
        print(f'>Accuracy: real = {acc_real}, fake = {acc_fake}')       
