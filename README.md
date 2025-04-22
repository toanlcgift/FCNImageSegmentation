# FCNImageSegmentation

Fully-Convolutional Network (FCN) implementation for image segmentation tasks using TensorFlow and Keras. This repository demonstrates the application of FCNs on the Oxford-IIIT Pets dataset for semantic segmentation.

## Introduction

This project implements Fully-Convolutional Networks (FCNs) for image segmentation, based on the paper [Fully Convolutional Networks for Semantic Segmentation by Long et. al. (2014)](https://arxiv.org/abs/1411.4038). FCNs extend image classification problems to pixel-wise classification tasks.

### Key Features:
- **Backbone:** Uses VGG-19 as the base feature extractor.
- **Versions:** Implements three FCN variants:
  - FCN-32S
  - FCN-16S
  - FCN-8S
- **Dataset:** Oxford-IIIT Pets dataset containing 7,349 images and segmentation masks.

![FCN Architecture](https://i.imgur.com/Ttros06.png)
*Diagram 1: FCN Architecture Overview*

## Installation

### Prerequisites:
- Python 3.8+
- TensorFlow
- Keras
- Other dependencies listed in `requirements.txt` (if available).

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/toanlcgift/FCNImageSegmentation.git
   cd FCNImageSegmentation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Dataset Preparation
The project uses the Oxford-IIIT Pets dataset, which can be automatically downloaded using TensorFlow datasets (`tensorflow_datasets`).

### Running the Code
Run the `application.py` script to train and evaluate the FCN models:
```bash
python application.py
```

### Key Parameters
- Input size: `(224, 224)`
- Batch size: `32`
- Classes: `4`
- Optimizer: AdamW
- Epochs: `20`

### Model Training
- **FCN-32S**: Upsamples the final layer by a factor of 32.
- **FCN-16S**: Combines outputs from the final and penultimate layers, upsampled by a factor of 16.
- **FCN-8S**: Combines outputs from the final three layers, upsampled by a factor of 8.

### Visualization
The script provides visualization for:
1. Dataset samples with segmentation masks.
2. Training metrics (accuracy, loss, mIOU) for comparative analysis of FCN variants.
3. Predicted segmentation masks from each FCN model.

## Results

### Training Metrics:
- **Accuracy**: Pixel-wise classification accuracy.
- **Mean Intersection over Union (mIOU)**: Evaluates segmentation performance.

### Sample Output:
![Segmentation Visualization](https://i.imgur.com/PerTKjf.png)
*Diagram 2: Segmentation Results*

## Acknowledgements

Special thanks to:
- [Suvaditya Mukherjee](https://twitter.com/halcyonrayes) for initial implementation.
- [Google Developer Experts](https://developers.google.com/community/experts) for support.

## References
- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- [Hugging Face Image Segmentation Models](https://huggingface.co/models?pipeline_tag=image-segmentation)
- [PyImageSearch Blog on Semantic Segmentation](https://pyimagesearch.com/2018/09/03/semantic-segmentation-with-opencv-and-deep-learning/)
