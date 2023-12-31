#https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 
#https://storage.googleapis.com/tensorflow-1-public/course2/cats_and_dogs_filtered.zip

import tensorflow as tf
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.93):
      print("\nReached 93.0% accuracy so cancelling training!")
      self.model.stop_training = True

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
import os


current_dir = os.getcwd()
print(current_dir)

# Set the weights file you downloaded into a variable
#data_path = os.path.join(current_dir, "tmp\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")
data_path = os.path.join(current_dir, "tmp\inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
#local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
current_dir = os.getcwd()
print(data_path)


# Initialize the base model.
# Set the input shape and remove the dense layers.
#pre_trained_model = InceptionV3(input_shape = (150, 150, 3),
#                                include_top = False,
#                                weights = None)
pre_trained_model = InceptionV3(include_top=True, weights='imagenet')

# Load the pre-trained weights you downloaded.
pre_trained_model.load_weights(data_path)

# Freeze the weights of the layers.
for layer in pre_trained_model.layers:
  layer.trainable = False


pre_trained_model.summary()

# Choose `mixed7` as the last layer of your base model
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)


################
import numpy as np
import matplotlib.pyplot as plt
# The weights are part of the layer's trainable parameters
some_filter_with_weight = pre_trained_model.get_layer('conv2d_1')
weights = some_filter_with_weight.get_weights()

# The weights include the filter weights and biases
filter_weights = weights[0]  # Filter weights
#biases = weights[1] 

# Plot the first filter weights
first_filter_weights = filter_weights[:, :, 0, 8]

# Plot the first filter weights
plt.imshow(first_filter_weights, cmap='viridis', interpolation='none', aspect='auto')
plt.title('First Filter Weights (Channel 0)')
plt.colorbar()
plt.show()
########



last_output = last_layer.output

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)

# Append the dense network to the base model
model = Model(pre_trained_model.input, x)

# Print the model summary. See your dense network connected at the end.
model.summary()

##################
# Showing features after mixed7 layer
#import numpy as np
#import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Create a new model that outputs the activations of the 'mixed7' layer
activation_model = Model(inputs=pre_trained_model.input, outputs=last_output)

# Choose an image for visualization
img_path = 'Ava.jpg'  # Replace with the path to an image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Get the activations of the 'mixed7' layer for the chosen image
activations = activation_model.predict(img_array)

# Print the shape of the activations
print("Shape of 'mixed7' layer activations:", activations.shape)

# Visualize the activations (example for the first channel)
i = 1
plt.matshow(activations[0, :, :, i], cmap='viridis')
plt.show()


#######

# Set the training parameters
model.compile(optimizer = RMSprop(learning_rate=0.0001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

data_path = os.path.join(current_dir, "tmp\cats_and_dogs_filtered.zip")

import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Extract the archive
zip_ref = zipfile.ZipFile(data_path, 'r')
zip_ref.extractall("tmp/")
zip_ref.close()

# Define our example directories and files
base_dir = 'tmp/cats_and_dogs_filtered'

train_dir = os.path.join( base_dir, 'train')
validation_dir = os.path.join( base_dir, 'validation')

# Directory with training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')

# Directory with training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')

# Directory with validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20,
                                                    class_mode = 'binary',
                                                    target_size = (150, 150))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory( validation_dir,
                                                          batch_size  = 20,
                                                          class_mode  = 'binary',
                                                          target_size = (150, 150))



# Train the model.
callbacks = myCallback()
history = model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 100,
            epochs = 20,
            validation_steps = 50,
            verbose = 2,
            callbacks=callbacks)

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()



