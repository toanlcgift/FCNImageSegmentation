using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras;

namespace ConsoleApp
{
    public static class ImageNetUtils
    {
        private static Dictionary<string, (string, string)> ClassIndex;
        private const string ClassIndexPath = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json";

        public static float[,] PreprocessInput(float[,] x, string dataFormat = null, string mode = "caffe")
        {
            if (mode != "caffe" && mode != "tf" && mode != "torch")
                throw new ArgumentException($"Expected mode to be one of `caffe`, `tf` or `torch`. Received: mode={mode}");

            dataFormat ??= Backend.ImageDataFormat();
            if (dataFormat != "channels_first" && dataFormat != "channels_last")
                throw new ArgumentException($"Expected data_format to be one of `channels_first` or `channels_last`. Received: data_format={dataFormat}");

            return PreprocessArray(x, dataFormat, mode);
        }

        public static List<List<(string ClassName, string ClassDescription, float Score)>> DecodePredictions(float[,] preds, int top = 5)
        {
            if (preds.GetLength(1) != 1000)
                throw new ArgumentException($"`decode_predictions` expects a batch of predictions (i.e. a 2D array of shape (samples, 1000)). Received array with shape: {preds.GetLength(0)}, {preds.GetLength(1)}");

            if (ClassIndex == null)
            {
                var json = FileUtils.GetFile("imagenet_class_index.json", ClassIndexPath, "models", "c2c37ea517e94d9795004a39431a14cb");
                ClassIndex = JsonConvert.DeserializeObject<Dictionary<string, (string, string)>>(File.ReadAllText(json));
            }

            var results = new List<List<(string, string, float)>>();
            for (int i = 0; i < preds.GetLength(0); i++)
            {
                var pred = Enumerable.Range(0, preds.GetLength(1)).Select(j => preds[i, j]).ToArray();
                var topIndices = pred.Select((value, index) => (value, index)).OrderByDescending(x => x.value).Take(top).Select(x => x.index).ToArray();
                var result = topIndices.Select(index => (ClassIndex[index.ToString()].Item1, ClassIndex[index.ToString()].Item2, pred[index])).ToList();
                results.Add(result);
            }

            return results;
        }

        private static float[,] PreprocessArray(float[,] x, string dataFormat, string mode)
        {
            if (mode == "tf")
            {
                for (int i = 0; i < x.GetLength(0); i++)
                {
                    for (int j = 0; j < x.GetLength(1); j++)
                    {
                        x[i, j] /= 127.5f;
                        x[i, j] -= 1.0f;
                    }
                }
                return x;
            }
            else if (mode == "torch")
            {
                for (int i = 0; i < x.GetLength(0); i++)
                {
                    for (int j = 0; j < x.GetLength(1); j++)
                    {
                        x[i, j] /= 255.0f;
                    }
                }
                // Normalization logic for "torch" mode can be added here.
            }
            else
            {
                // "caffe" mode preprocessing logic.
                // Conversion from RGB to BGR and zero-centering by mean pixel.
            }

            return x;
        }

        public static int[] ObtainInputShape(
        int[] inputShape,
        int defaultSize,
        int minSize,
        string dataFormat,
        bool requireFlatten,
        string weights = null
    )
        {
            if (weights != "imagenet" && inputShape != null && inputShape.Length == 3)
            {
                if (dataFormat == "channels_first")
                {
                    if (inputShape[0] != 1 && inputShape[0] != 3)
                    {
                        Console.WriteLine($"Warning: This model usually expects 1 or 3 input channels. However, it was passed an inputShape with {inputShape[0]} input channels.");
                    }
                    return new[] { inputShape[0], defaultSize, defaultSize };
                }
                else
                {
                    if (inputShape[2] != 1 && inputShape[2] != 3)
                    {
                        Console.WriteLine($"Warning: This model usually expects 1 or 3 input channels. However, it was passed an inputShape with {inputShape[2]} input channels.");
                    }
                    return new[] { defaultSize, defaultSize, inputShape[2] };
                }
            }
            else
            {
                if (dataFormat == "channels_first")
                {
                    return new[] { 3, defaultSize, defaultSize };
                }
                else
                {
                    return new[] { defaultSize, defaultSize, 3 };
                }
            }

            if (weights == "imagenet" && requireFlatten)
            {
                if (inputShape != null)
                {
                    var defaultShape = dataFormat == "channels_first"
                        ? new[] { 3, defaultSize, defaultSize }
                        : new[] { defaultSize, defaultSize, 3 };

                    if (!inputShape.SequenceEqual(defaultShape))
                    {
                        throw new ArgumentException($"When setting `includeTop=true` and loading `imagenet` weights, `inputShape` should be {string.Join(", ", defaultShape)}. Received: inputShape={string.Join(", ", inputShape)}");
                    }
                }
                return inputShape;
            }

            if (inputShape != null)
            {
                if (dataFormat == "channels_first")
                {
                    if (inputShape.Length != 3)
                    {
                        throw new ArgumentException("`inputShape` must be a tuple of three integers.");
                    }
                    if (inputShape[0] != 3 && weights == "imagenet")
                    {
                        throw new ArgumentException($"The input must have 3 channels; Received `inputShape={string.Join(", ", inputShape)}`");
                    }
                    if ((inputShape[1] != 0 && inputShape[1] < minSize) || (inputShape[2] != 0 && inputShape[2] < minSize))
                    {
                        throw new ArgumentException($"Input size must be at least {minSize}x{minSize}; Received: inputShape={string.Join(", ", inputShape)}");
                    }
                }
                else
                {
                    if (inputShape.Length != 3)
                    {
                        throw new ArgumentException("`inputShape` must be a tuple of three integers.");
                    }
                    if (inputShape[2] != 3 && weights == "imagenet")
                    {
                        throw new ArgumentException($"The input must have 3 channels; Received `inputShape={string.Join(", ", inputShape)}`");
                    }
                    if ((inputShape[0] != 0 && inputShape[0] < minSize) || (inputShape[1] != 0 && inputShape[1] < minSize))
                    {
                        throw new ArgumentException($"Input size must be at least {minSize}x{minSize}; Received: inputShape={string.Join(", ", inputShape)}");
                    }
                }
            }
            else
            {
                if (requireFlatten)
                {
                    return dataFormat == "channels_first"
                        ? new[] { 3, defaultSize, defaultSize }
                        : new[] { defaultSize, defaultSize, 3 };
                }
                else
                {
                    return dataFormat == "channels_first"
                        ? new[] { 3, -1, -1 }
                        : new[] { -1, -1, 3 };
                }
            }

            if (requireFlatten && inputShape.Contains(-1))
            {
                throw new ArgumentException($"If `includeTop=true`, you should specify a static `inputShape`. Received: inputShape={string.Join(", ", inputShape)}");
            }

            return inputShape;
        }

        public static void ValidateActivation(string classifierActivation, string weights)
        {
            // Validates that the classifierActivation is compatible with the weights.  

            if (weights == null)
            {
                return;
            }

            var activation = KerasApi.keras.activations.GetActivationFromName(classifierActivation);
            var allowedActivations = new HashSet<object>
       {
           KerasApi.keras.activations.GetActivationFromName("softmax"),
           KerasApi.keras.activations.GetActivationFromName(null)
       };

            if (!allowedActivations.Contains(activation))
            {
                throw new ArgumentException(
                    $"Only `null` and `softmax` activations are allowed for the `classifierActivation` argument when using " +
                    $"pretrained weights, with `includeTop=true`; Received: classifierActivation={classifierActivation}"
                );
            }
        }
    }

    public static class Backend
    {
        public static string ImageDataFormat()
        {
            // Return the global image data format, e.g., "channels_last".
            return "channels_last";
        }
    }

    public static class FileUtils
    {
        public static string GetFile(string fileName, string url, string cacheSubdir, string fileHash)
        {
            // Logic to download and cache the file locally.
            return Path.Combine(cacheSubdir, fileName);
        }
    }
}
