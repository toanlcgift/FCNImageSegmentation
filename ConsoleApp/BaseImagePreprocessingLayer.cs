using Keras.Src.Layers.Preprocessing;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras.Saving;
using Tensorflow.NumPy;

namespace ConsoleApp
{
    public abstract class BaseImagePreprocessingLayer : TFDataLayer
    {
        private static readonly bool _USE_BASE_FACTOR = true;
        private static readonly (double, double) _FACTOR_BOUNDS = (-1, 1);

        public (double, double) Factor { get; private set; }
        public string BoundingBoxFormat { get; private set; }
        public string DataFormat { get; private set; }

        protected BaseImagePreprocessingLayer(
            double? factor = null,
            string boundingBoxFormat = null,
            string dataFormat = null,
            Dictionary<string, object> kwargs = null
        ) : base(new Tensorflow.Keras.ArgsDefinition.LayerArgs() { })
        {
            BoundingBoxFormat = boundingBoxFormat;
            DataFormat = BackendConfig.StandardizeDataFormat(dataFormat);

            if (_USE_BASE_FACTOR)
            {
                factor ??= 0.0;
                SetFactor(factor.Value);
            }
            else if (factor.HasValue)
            {
                throw new ArgumentException(
                    $"Layer {GetType().Name} does not take a `factor` argument. Received: factor={factor}"
                );
            }
        }

        private void SetFactor(double factor)
        {
            string errorMsg = $"The `factor` argument should be a number (or a list of two numbers) in the range [{_FACTOR_BOUNDS.Item1}, {_FACTOR_BOUNDS.Item2}]. Received: factor={factor}";

            if (factor < _FACTOR_BOUNDS.Item1 || factor > _FACTOR_BOUNDS.Item2)
            {
                throw new ArgumentException(errorMsg);
            }

            double lower = Math.Max(-Math.Abs(factor), _FACTOR_BOUNDS.Item1);
            double upper = Math.Abs(factor);
            Factor = (lower, upper);
        }

        public virtual object GetRandomTransformation(object data, bool training = true, int? seed = null)
        {
            return null;
        }

        public abstract Tensor TransformImages(Tensor images, object transformation, bool training = true);

        public abstract Tensor TransformLabels(object labels, object transformation, bool training = true);

        public abstract Tensor TransformBoundingBoxes(object boundingBoxes, object transformation, bool training = true);

        public abstract Tensor TransformSegmentationMasks(object segmentationMasks, object transformation, bool training = true);

        public Tensor TransformSingleImage(Tensor image, object transformation, bool training = true)
        {
            
            var images = Backend.Numpy.ExpandDims(image, axis: 0);
            var outputs = TransformImages(images, transformation, training);
            return Backend.Numpy.Squeeze(outputs, axis: 0);
        }

        public Tensor TransformSingleLabel(object label, object transformation, bool training = true)
        {
            var labels = Backend.Numpy.ExpandDims(label, axis: 0);
            var outputs = TransformLabels(labels, transformation, training);
            return Backend.Numpy.Squeeze(outputs, axis: 0);
        }

        public Tensor TransformSingleBoundingBox(object boundingBox, object transformation, bool training = true)
        {
            var boundingBoxes = FormatSingleInputBoundingBox(boundingBox);
            var outputs = TransformBoundingBoxes(boundingBoxes, transformation, training);
            return FormatSingleOutputBoundingBox(outputs);
        }

        public Tensor TransformSingleSegmentationMask(object segmentationMask, object transformation, bool training = true)
        {
            var segmentationMasks = Backend.Numpy.ExpandDims(segmentationMask, axis: 0);
            var outputs = TransformSegmentationMasks(segmentationMasks, transformation, training);
            return Backend.Numpy.Squeeze(outputs, axis: 0);
        }

        private bool IsBatched(object maybeImageBatch)
        {
            var shape = Backend.Core.Shape(maybeImageBatch);
            if (shape.Length == 3) return false;
            if (shape.Length == 4) return true;

            throw new ArgumentException($"Expected image tensor to have rank 3 (single image) or 4 (batch of images). Received: data.shape={shape}");
        }

        public object Call(object data, bool training = true)
        {
            var transformation = GetRandomTransformation(data, training);

            if (data is Dictionary<string, object> dictData)
            {
                bool isBatched = IsBatched(dictData["images"]);
                dictData["images"] = isBatched
                    ? TransformImages(Backend.ConvertToTensor(dictData["images"]), transformation, training)
                    : TransformSingleImage(Backend.ConvertToTensor(dictData["images"]), transformation, training);

                if (dictData.ContainsKey("bounding_boxes"))
                {
                    if (BoundingBoxFormat == null)
                    {
                        throw new ArgumentException($"You passed an input with a 'bounding_boxes' key, but you didn't specify a bounding box format. Pass a `bounding_box_format` argument to your {GetType().Name} layer, e.g. `bounding_box_format='xyxy'`.");
                    }

                    var boundingBoxes = DensifyBoundingBoxes(dictData["bounding_boxes"], isBatched, Backend);

                    dictData["bounding_boxes"] = isBatched
                        ? TransformBoundingBoxes(boundingBoxes, transformation, training)
                        : TransformSingleBoundingBox(boundingBoxes, transformation, training);
                }

                if (dictData.ContainsKey("labels"))
                {
                    dictData["labels"] = isBatched
                        ? TransformLabels(Backend.ConvertToTensor(dictData["labels"]), transformation, training)
                        : TransformSingleLabel(Backend.ConvertToTensor(dictData["labels"]), transformation, training);
                }

                if (dictData.ContainsKey("segmentation_masks"))
                {
                    dictData["segmentation_masks"] = isBatched
                        ? TransformSegmentationMasks(dictData["segmentation_masks"], transformation, training)
                        : TransformSingleSegmentationMask(dictData["segmentation_masks"], transformation, training);
                }

                return dictData;
            }

            // `data` is just images.
            return IsBatched(data)
                ? TransformImages(Backend.ConvertToTensor(data), transformation, training)
                : TransformSingleImage(Backend.ConvertToTensor(data), transformation, training);
        }

        private object FormatSingleInputBoundingBox(object boundingBox)
        {
            // Implementation for formatting single input bounding box
            throw new NotImplementedException();
        }

        private object FormatSingleOutputBoundingBox(object boundingBoxes)
        {
            // Implementation for formatting single output bounding box
            throw new NotImplementedException();
        }

        public override IKerasConfig get_config()
        {
            var config = base.get_config();
            if (BoundingBoxFormat != null)
            {
                //config["bounding_box_format"] = BoundingBoxFormat;
            }
            return config;
        }
    }
}
