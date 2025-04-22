using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras.Saving;

namespace ConsoleApp
{
    public class Resizing : BaseImagePreprocessingLayer
    {
        public int Height { get; }
        public int Width { get; }
        public string Interpolation { get; }
        public bool CropToAspectRatio { get; }
        public bool PadToAspectRatio { get; }
        public string FillMode { get; }
        public float FillValue { get; }
        public bool Antialias { get; }
        public string DataFormat { get; }
        private int HeightAxis { get; }
        private int WidthAxis { get; }

        public Resizing(
            int height,
            int width,
            string interpolation = "bilinear",
            bool cropToAspectRatio = false,
            bool padToAspectRatio = false,
            string fillMode = "constant",
            float fillValue = 0.0f,
            bool antialias = false,
            string dataFormat = null
        )
        {
            Height = height;
            Width = width;
            Interpolation = interpolation;
            CropToAspectRatio = cropToAspectRatio;
            PadToAspectRatio = padToAspectRatio;
            FillMode = fillMode;
            FillValue = fillValue;
            Antialias = antialias;
            DataFormat = StandardizeDataFormat(dataFormat);

            if (DataFormat == "channels_first")
            {
                HeightAxis = -2;
                WidthAxis = -1;
            }
            else if (DataFormat == "channels_last")
            {
                HeightAxis = -3;
                WidthAxis = -2;
            }
        }

        public Tensor TransformImages(Tensor images, object transformation = null, bool training = true)
        {
            var size = (Height, Width);
            var resized = Backend.Image.Resize(
                images,
                size,
                Interpolation,
                Antialias,
                DataFormat,
                CropToAspectRatio,
                PadToAspectRatio,
                FillMode,
                FillValue
            );

            if (resized.DType == images.DType)
                return resized;

            if (Backend.IsIntDType(images.DType))
                resized = Backend.Numpy.Round(resized);

            return SaturateCast(resized, images.DType, Backend);
        }

        public Tensor TransformSegmentationMasks(Tensor segmentationMasks, object transformation = null, bool training = true)
        {
            return TransformImages(segmentationMasks);
        }

        public Tensor TransformLabels(Tensor labels, object transformation = null, bool training = true)
        {
            return labels;
        }

        public (int, int) GetRandomTransformation(object data, bool training = true, object seed = null)
        {
            var inputShape = data is Dictionary<string, Tensor> dict
                ? Backend.Shape(dict["images"])
                : Backend.Shape(data);

            var inputHeight = inputShape[HeightAxis];
            var inputWidth = inputShape[WidthAxis];

            return (inputHeight, inputWidth);
        }

        public BoundingBox TransformBoundingBoxes(BoundingBox boundingBoxes, (int, int) transformation, bool training = true)
        {
            var (inputHeight, inputWidth) = transformation;
            var ops = Backend;

            // Logic for transforming bounding boxes goes here.  

            return boundingBoxes;
        }

        public override Shape ComputeOutputShape(Shape inputShape)
        {
            var shape = inputShape.ToList();

            if (shape.Count == 4)
            {
                if (DataFormat == "channels_last")
                {
                    shape[1] = Height;
                    shape[2] = Width;
                }
                else
                {
                    shape[2] = Height;
                    shape[3] = Width;
                }
            }
            else
            {
                if (DataFormat == "channels_last")
                {
                    shape[0] = Height;
                    shape[1] = Width;
                }
                else
                {
                    shape[1] = Height;
                    shape[2] = Width;
                }
            }

            return new Shape(shape);
        }

        public override  IKerasConfig get_config()
        {
            // var baseConfig = base.get_config();
            // var config = new Dictionary<string, object>
            //{
            //    { "height", Height },
            //    { "width", Width },
            //    { "interpolation", Interpolation },
            //    { "crop_to_aspect_ratio", CropToAspectRatio },
            //    { "pad_to_aspect_ratio", PadToAspectRatio },
            //    { "fill_mode", FillMode },
            //    { "fill_value", FillValue },
            //    { "antialias", Antialias },
            //    { "data_format", DataFormat }
            //};

            // foreach (var kvp in baseConfig)
            //     config[kvp.Key] = kvp.Value;

            // return config;
            return base.get_config();
        }

        private string StandardizeDataFormat(string dataFormat)
        {
            // Logic to standardize data format.  
            return dataFormat ?? "channels_last";
        }

        private Tensor SaturateCast(Tensor tensor, Type targetType, Backend backend)
        {
            // Logic for saturate casting.  
            return tensor;
        }
    }
}
