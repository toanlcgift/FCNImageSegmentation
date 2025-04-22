using Microsoft.VisualBasic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;

namespace ConsoleApp
{
    public static class VGG19
    {
        private const string WEIGHTS_PATH = "https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5";
        private const string WEIGHTS_PATH_NO_TOP = "https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5";

        public static Functional BuildVGG19(
            bool includeTop = true,
            string weights = "imagenet",
            Tensor inputTensor = null,
            int[] inputShape = null,
            string pooling = null,
            int classes = 1000,
            string classifierActivation = "softmax",
            string name = "vgg19")
        {
            if (!(weights == "imagenet" || weights == null || File.Exists(weights)))
            {
                throw new ArgumentException("The `weights` argument should be either `None` (random initialization), 'imagenet' (pre-training on ImageNet), or the path to the weights file to be loaded.");
            }

            if (weights == "imagenet" && includeTop && classes != 1000)
            {
                throw new ArgumentException("If using `weights='imagenet'` with `includeTop=True`, `classes` should be 1000.");
            }

            // Determine proper input shape
            var inputShapeResolved = ImageNetUtils.ObtainInputShape(
                inputShape,
                defaultSize: 224,
                minSize: 32,
                dataFormat: K.ImageDataFormat(),
                requireFlatten: includeTop,
                weights: weights
            );

            Tensor imgInput;
            if (inputTensor == null)
            {
                imgInput = K.Input(shape: inputShapeResolved);
            }
            else
            {
                imgInput = inputTensor;
            }

            // Block 1
            var x = new Conv2D(new Tensorflow.Keras.ArgsDefinition.Conv2DArgs()
            {
                Filters = 64,
                KernelSize = new Shape(3, 3),
                Activation = KerasApi.keras.activations.Relu,
                Padding = "same",
                Name = "block1_conv1"
            }).Apply(imgInput);
            x = new Conv2D(new Tensorflow.Keras.ArgsDefinition.Conv2DArgs() { Filters = 64, KernelSize = new Shape(3, 3), Activation = KerasApi.keras.activations.Relu, Padding = "same", Name = "block1_conv2" }).Apply(x);
            x = new MaxPooling2D(new Tensorflow.Keras.ArgsDefinition.MaxPooling2DArgs() { PoolSize = (2, 2), Strides = (2, 2), Name = "block1_pool" }).Apply(x);

            // Block 2
            x = new Conv2D(new Tensorflow.Keras.ArgsDefinition.Conv2DArgs() { Filters = 128, KernelSize = new Shape(3, 3), Activation = KerasApi.keras.activations.Relu, Padding = "same", Name = "block2_conv1" }).Apply(x);
            x = new Conv2D(new Tensorflow.Keras.ArgsDefinition.Conv2DArgs() { Filters = 128, KernelSize = new Shape(3, 3), Activation = KerasApi.keras.activations.Relu, Padding = "same", Name = "block2_conv2" }).Apply(x);
            x = new MaxPooling2D(new Tensorflow.Keras.ArgsDefinition.MaxPooling2DArgs() { PoolSize = (2, 2), Strides = (2, 2), Name = "block2_pool" }).Apply(x);

            // Block 3
            x = new Conv2D(new Tensorflow.Keras.ArgsDefinition.Conv2DArgs() { Filters = 256, KernelSize = new Shape(3, 3), Activation = KerasApi.keras.activations.Relu, Padding = "same", Name = "block3_conv1" }).Apply(x);
            x = new Conv2D(new Tensorflow.Keras.ArgsDefinition.Conv2DArgs() { Filters = 256, KernelSize = new Shape(3, 3), Activation = KerasApi.keras.activations.Relu, Padding = "same", Name = "block3_conv2" }).Apply(x);
            x = new Conv2D(new Tensorflow.Keras.ArgsDefinition.Conv2DArgs() { Filters = 256, KernelSize = new Shape(3, 3), Activation = KerasApi.keras.activations.Relu, Padding = "same", Name = "block3_conv3" }).Apply(x);
            x = new Conv2D(new Tensorflow.Keras.ArgsDefinition.Conv2DArgs() { Filters = 256, KernelSize = new Shape(3, 3), Activation = KerasApi.keras.activations.Relu, Padding = "same", Name = "block3_conv4" }).Apply(x);
            x = new MaxPooling2D(new Tensorflow.Keras.ArgsDefinition.MaxPooling2DArgs() { PoolSize = (2, 2), Strides = (2, 2), Name = "block3_pool" }).Apply(x);

            // Block 4
            x = new Conv2D(new Tensorflow.Keras.ArgsDefinition.Conv2DArgs() { Filters = 512, KernelSize = new Shape(3, 3), Activation = KerasApi.keras.activations.Relu, Padding = "same", Name = "block4_conv1" }).Apply(x);
            x = new Conv2D(new Tensorflow.Keras.ArgsDefinition.Conv2DArgs() { Filters = 512, KernelSize = new Shape(3, 3), Activation = KerasApi.keras.activations.Relu, Padding = "same", Name = "block4_conv2" }).Apply(x);
            x = new Conv2D(new Tensorflow.Keras.ArgsDefinition.Conv2DArgs() { Filters = 512, KernelSize = new Shape(3, 3), Activation = KerasApi.keras.activations.Relu, Padding = "same", Name = "block4_conv3" }).Apply(x);
            x = new Conv2D(new Tensorflow.Keras.ArgsDefinition.Conv2DArgs() { Filters = 512, KernelSize = new Shape(3, 3), Activation = KerasApi.keras.activations.Relu, Padding = "same", Name = "block4_conv4" }).Apply(x);
            x = new MaxPooling2D(new Tensorflow.Keras.ArgsDefinition.MaxPooling2DArgs() { PoolSize = (2, 2), Strides = (2, 2), Name = "block4_pool" }).Apply(x);

            // Block 5
            x = new Conv2D(new Tensorflow.Keras.ArgsDefinition.Conv2DArgs() { Filters = 512, KernelSize = new Shape(3, 3), Activation = KerasApi.keras.activations.Relu, Padding = "same", Name = "block5_conv1" }).Apply(x);
            x = new Conv2D(new Tensorflow.Keras.ArgsDefinition.Conv2DArgs() { Filters = 512, KernelSize = new Shape(3, 3), Activation = KerasApi.keras.activations.Relu, Padding = "same", Name = "block5_conv2" }).Apply(x);
            x = new Conv2D(new Tensorflow.Keras.ArgsDefinition.Conv2DArgs() { Filters = 512, KernelSize = new Shape(3, 3), Activation = KerasApi.keras.activations.Relu, Padding = "same", Name = "block5_conv3" }).Apply(x);
            x = new Conv2D(new Tensorflow.Keras.ArgsDefinition.Conv2DArgs() { Filters = 512, KernelSize = new Shape(3, 3), Activation = KerasApi.keras.activations.Relu, Padding = "same", Name = "block5_conv4" }).Apply(x);
            x = new MaxPooling2D(new Tensorflow.Keras.ArgsDefinition.MaxPooling2DArgs() { PoolSize = (2, 2), Strides = (2, 2), Name = "block5_pool" }).Apply(x);

            if (includeTop)
            {
                // Classification block
                x = new Flatten(new Tensorflow.Keras.ArgsDefinition.FlattenArgs() { Name = "flatten" }).Apply(x);
                x = new Dense(new Tensorflow.Keras.ArgsDefinition.DenseArgs() { Units = 4096, Activation = KerasApi.keras.activations.Relu, Name = "fc1" }).Apply(x);
                x = new Dense(new Tensorflow.Keras.ArgsDefinition.DenseArgs() { Units = 4096, Activation = KerasApi.keras.activations.Relu, Name = "fc2" }).Apply(x);
                ImageNetUtils.ValidateActivation(classifierActivation, weights);
                x = new Dense(new Tensorflow.Keras.ArgsDefinition.DenseArgs() { Units = classes, Activation = KerasApi.keras.activations.Softmax, Name = "predictions" }).Apply(x);
            }
            else
            {
                if (pooling == "avg")
                {
                    x = new GlobalAveragePooling2D(new Tensorflow.Keras.ArgsDefinition.Pooling2DArgs()).Apply(x);
                }
                else if (pooling == "max")
                {
                    x = new GlobalMaxPooling2D(new Tensorflow.Keras.ArgsDefinition.Pooling2DArgs()).Apply(x);
                }
            }

            // Create model
            var model = new Functional(imgInput, x, name: name);

            // Load weights
            if (weights == "imagenet")
            {
                string weightsPath = includeTop
                    ? FileUtils.GetFile("vgg19_weights_tf_dim_ordering_tf_kernels.h5", WEIGHTS_PATH, cacheSubdir: "models", fileHash: "cbe5617147190e668d6c5d5026f83318")
                    : FileUtils.GetFile("vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5", WEIGHTS_PATH_NO_TOP, cacheSubdir: "models", fileHash: "253f8cb515780f3b799900260a226db6");
                model.LoadWeights(weightsPath);
            }
            else if (weights != null)
            {
                model.LoadWeights(weights);
            }

            return model;
        }
    }
}
