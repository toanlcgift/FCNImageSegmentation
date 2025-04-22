using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
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
            var x = new Conv2D64, (3, 3), activation: "relu", padding: "same", name: "block1_conv1").Apply(imgInput);
            x = new Conv2D(64, (3, 3), activation: "relu", padding: "same", name: "block1_conv2").Apply(x);
            x = new MaxPooling2D((2, 2), strides: (2, 2), name: "block1_pool").Apply(x);

            // Block 2
            x = new Conv2D(128, (3, 3), activation: "relu", padding: "same", name: "block2_conv1").Apply(x);
            x = new Conv2D(128, (3, 3), activation: "relu", padding: "same", name: "block2_conv2").Apply(x);
            x = new MaxPooling2D((2, 2), strides: (2, 2), name: "block2_pool").Apply(x);

            // Block 3
            x = new Conv2D(256, (3, 3), activation: "relu", padding: "same", name: "block3_conv1").Apply(x);
            x = new Conv2D(256, (3, 3), activation: "relu", padding: "same", name: "block3_conv2").Apply(x);
            x = new Conv2D(256, (3, 3), activation: "relu", padding: "same", name: "block3_conv3").Apply(x);
            x = new Conv2D(256, (3, 3), activation: "relu", padding: "same", name: "block3_conv4").Apply(x);
            x = new MaxPooling2D((2, 2), strides: (2, 2), name: "block3_pool").Apply(x);

            // Block 4
            x = new Conv2D(512, (3, 3), activation: "relu", padding: "same", name: "block4_conv1").Apply(x);
            x = new Conv2D(512, (3, 3), activation: "relu", padding: "same", name: "block4_conv2").Apply(x);
            x = new Conv2D(512, (3, 3), activation: "relu", padding: "same", name: "block4_conv3").Apply(x);
            x = new Conv2D(512, (3, 3), activation: "relu", padding: "same", name: "block4_conv4").Apply(x);
            x = new MaxPooling2D((2, 2), strides: (2, 2), name: "block4_pool").Apply(x);

            // Block 5
            x = new Conv2D(512, (3, 3), activation: "relu", padding: "same", name: "block5_conv1").Apply(x);
            x = new Conv2D(512, (3, 3), activation: "relu", padding: "same", name: "block5_conv2").Apply(x);
            x = new Conv2D(512, (3, 3), activation: "relu", padding: "same", name: "block5_conv3").Apply(x);
            x = new Conv2D(512, (3, 3), activation: "relu", padding: "same", name: "block5_conv4").Apply(x);
            x = new MaxPooling2D((2, 2), strides: (2, 2), name: "block5_pool").Apply(x);

            if (includeTop)
            {
                // Classification block
                x = new Flatten(name: "flatten").Apply(x);
                x = new Dense(4096, activation: "relu", name: "fc1").Apply(x);
                x = new Dense(4096, activation: "relu", name: "fc2").Apply(x);
                ImageNetUtils.ValidateActivation(classifierActivation, weights);
                x = new Dense(classes, activation: classifierActivation, name: "predictions").Apply(x);
            }
            else
            {
                if (pooling == "avg")
                {
                    x = new GlobalAveragePooling2D().Apply(x);
                }
                else if (pooling == "max")
                {
                    x = new GlobalMaxPooling2D().Apply(x);
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
