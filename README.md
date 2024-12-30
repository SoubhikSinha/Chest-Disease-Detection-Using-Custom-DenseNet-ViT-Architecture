# Chest Disease Detection Using Custom DenseNet-ViT (Hybrid) Architecture

<br>
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/SoubhikSinha/Chest-Disease-Detection-Using-Custom-DenseNet-ViT-Architecture/blob/main/Deploy-Test%20Images/4ceeb362df213ccf2c0b4d6388bba1_gallery.jpg" style="height: 200px; width: auto;">
  <img src="https://github.com/SoubhikSinha/Chest-Disease-Detection-Using-Custom-DenseNet-ViT-Architecture/blob/main/Deploy-Test%20Images/642x361_SLIDE_4_What_Does_Lung_Cancer_Look_Like.jpg?raw=true" style="height: 200px; width: auto;">
</div>


Acknowledgements
---
I would like to express my heartfelt gratitude to the creators of the datasets used in this project : `Paul Timothy Mooney` for the *Chest X-Ray Pneumonia Dataset*, `Tawsifur Rahman` for the *Tuberculosis Chest X-Ray Dataset*, and `Mohamed Hanyyy` for the *Chest CT-Scan Images Dataset*, all available on Kaggle. Their invaluable contributions provided the foundation for the development of this project. I also extend my sincere thanks to my peers and mentors for their guidance, support, and feedback, which played a crucial role in the success of this project.

<br>

About the Project
---
This project focuses on developing an AI-driven medical image analysis system aimed at diagnosing three major thoracic diseases : *[Pneumonia](https://medlineplus.gov/pneumonia.html), [Tuberculosis](https://www.who.int/news-room/fact-sheets/detail/tuberculosis#:~:text=Tuberculosis%20(TB)%20is%20an%20infectious,been%20infected%20with%20TB%20bacteria.), and [Lung cancer](https://medlineplus.gov/lungcancer.html#:~:text=Lung%20cancer%20is%20cancer%20that,differently%20and%20are%20treated%20differently.)*. By leveraging advanced deep learning techniques, the system processes radiological images, including chest [X-rays](https://www.nibib.nih.gov/science-education/science-topics/x-rays#:~:text=X%2Drays%20are%20a%20form,and%20structures%20inside%20the%20body.) and [CT scan](https://my.clevelandclinic.org/health/diagnostics/4808-ct-computed-tomography-scan) images, to classify pathologies and provide probability estimates for each condition. The tool enhances diagnostic accuracy and efficiency, supporting healthcare professionals in clinical decision-making and showcasing the potential for integrating AI into modern medical workflows. To make the system more accessible, a user-friendly application based on the *[Gradio](https://www.gradio.app/)* framework is developed, allowing users to upload chest imaging data and receive real-time predictions with associated probabilities. The application provides probabilistic predictions for pneumonia and tuberculosis, and multi-class probabilistic predictions for lung cancer, acting as an intuitive interface that bridges complex AI models and clinical practice for both medical professionals and researchers.

<br>

Datasets
---

 - [Chest X-Ray Images for Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) ▶️ Consists of 5,856 images, with 3,878 labeled as `infected` and 1,349 labeled as `normal`.
 - [Chest X-Ray Images for Tuberculosis](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) ▶️ Contains 4,200 images, of which 700 are labeled as `infected` and 3,500 as `normal`.
- [Chest CT-Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) ▶️ Comprises 1,000 CT-scan images, focusing on the classification of lung cancer types, specifically *[Adenocarcinoma](https://my.clevelandclinic.org/health/diseases/21652-adenocarcinoma-cancers), [Large Cell Carcinoma (LCC)](https://lcfamerica.org/about-lung-cancer/diagnosis/types/large-cell-carcinomas/), and [Squamous Cell Carcinoma (SCC)](https://www.skincancer.org/skin-cancer-information/squamous-cell-carcinoma/)*. 

<br>

Image Preprocessing Techniques
---
1.  **Histogram Equalization** ▶️ A technique that enhances the contrast of an image by redistributing its pixel intensity values, making the histogram more uniform.
    
2.  **CLAHE (Contrast Limited Adaptive Histogram Equalization)** : A variant of histogram equalization that operates locally on small regions (tiles) of the image to prevent over-amplification of noise in homogeneous areas.
    
3.  **Gaussian Blur** ▶️ A smoothing technique used to reduce noise and detail in an image by applying a Gaussian filter, which averages pixel values based on their distance from the center.
    
4.  **Resizing with Anti-Aliasing** ▶️ A process of changing an image's dimensions while using anti-aliasing techniques to smooth the image and reduce visual distortion or pixelation.
    
5.  **Edge Enhancement (Sharpening)** ▶️ A technique used to enhance the edges of objects in an image by amplifying high-frequency components, making them more distinct.
    
6.  **Intensity Normalization** ▶️ A method of adjusting the pixel values of an image to a specific range, typically [0, 1] or [0, 255], to standardize the image intensity for further processing.
    
7.  **Tensor Conversion and Normalization to [-1, 1] Range** ▶️ The process of converting image data into a tensor format and normalizing pixel values to a range of [-1, 1] for neural network compatibility and improved model performance.
    
8.  **Brightness and Contrast** ▶️ Techniques that adjust the overall brightness (lightness) and contrast (difference between light and dark areas) of an image to enhance visual clarity or correct lighting conditions.

<br>

DenseNet-ViT Hybrid Architecture
---
The `LightweightHybridDenseNetViT` architecture combines a [DenseNet](https://arxiv.org/abs/1608.06993) backbone with a simplified [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) module for tasks like pneumonia, tuberculosis, and lung cancer classification. Below is a detailed breakdown of each component of the architecture:

### 1. **DenseNet Backbone (`self.densenet`)** :

-   **DenseNet-121** : This is a pretrained model from the `torchvision` library, specifically [DenseNet121](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html). DenseNet is a type of convolutional neural network where each layer receives input from all previous layers, facilitating feature reuse and mitigating the vanishing gradient problem.
-   **Feature Extraction** : The DenseNet is used primarily as a feature extractor. The model is loaded without pretrained weights (`pretrained=False`), and it outputs a rich set of features which are passed to subsequent layers.

### 2. **Feature Channel Reduction (`self.conv_reduce`)** :

-   **1x1 Convolution Layer** : After passing through DenseNet, the feature maps are typically high-dimensional (e.g., 1024 channels). To reduce the number of channels, a 1x1 convolution (`self.conv_reduce`) is applied, which reduces the 1024 channels down to 64. This reduction helps in minimizing the computational complexity and is a necessary step before passing the feature maps to the transformer module.

### 3. **Simplified Vision Transformer (`self.vit`)** :

-   **Simplified ViT Module** : A Vision Transformer (ViT) is used for processing the feature maps from DenseNet. ViTs operate on sequences of patches extracted from an image and learn relationships between these patches. However, in this case, a **simplified ViT** module is employed to reduce the complexity compared to traditional ViTs.
    -   **TransformerEncoderLayer** : Each ViT block contains a `TransformerEncoderLayer`, which performs self-attention and then a feed-forward transformation (MLP - Multi-Level Perceptron). The ViT module here uses a smaller model with only 1 layer (`num_layers=1`), 2 attention heads (`num_heads=2`), and a smaller hidden dimension of 128 (`mlp_dim=128`).
    -   **Flattening and Reshaping** : Before passing the input feature map through the transformer, the input tensor is flattened to a sequence of tokens (patches) by using the `x.flatten(2).permute(2, 0, 1)` operation. This transforms the feature map from the shape `(B, C, H, W)` into a sequence that the transformer can process. After the transformer processes this sequence, the result is reshaped back into the spatial dimensions of the input.

### 4. **Global Average Pooling (GAP)** :

-   **Pooling across Spatial Dimensions** : The output from the ViT module is a set of feature maps with dimensions `(B, C, H, W)`. To summarize this spatial information, **Global Average Pooling (GAP)** is applied (`x.mean(dim=[2, 3])`). GAP calculates the average of each feature map across the spatial dimensions (`H` and `W`), resulting in a vector of size `(B, C)`, where `C` is the number of channels (64 in this case). This operation reduces the spatial resolution, allowing the model to focus on the global features in the image.

### 5. **Task-Specific Classification Heads** :

After the feature maps are pooled, the resulting vector is passed to three different fully connected (FC) layers for classification:

-   **Pneumonia Output (`self.fc_pneumonia`)** : A single output neuron with a sigmoid activation function, indicating a binary classification for pneumonia (0 or 1). The output is passed through the sigmoid activation to produce a probability.
-   **Tuberculosis Output (`self.fc_tuberculosis`)** : Similar to the pneumonia output, this FC layer also has a single output neuron for binary classification of tuberculosis.
-   **Lung Cancer Output (`self.fc_lung_cancer`)** : This output head has 4 output neurons, which corresponds to a multi-class classification task for lung cancer. The model outputs raw logits (scores for each class), which can be used for multi-class classification tasks.

### 6. **Forward Pass** :

-   **Input Processing** : The input image `x` is passed through DenseNet’s convolutional layers to extract feature maps.
-   **Feature Reduction** : A 1x1 convolution reduces the number of output channels.
-   **Transformer Processing** : The reduced feature maps are passed through the simplified Vision Transformer.
-   **Pooling** : Global Average Pooling is applied to the output of the transformer.
-   **Classification** : The pooled features are passed through the task-specific classification heads to produce predictions for pneumonia, tuberculosis, and lung cancer.

### Summary of Workflow :

1.  **DenseNet** extracts features from the input image.
2.  **Conv1x1 layer** reduces the number of output channels.
3.  **Simplified ViT** module processes the features, learning long-range dependencies between patches.
4.  **Global Average Pooling** aggregates spatial information.
5.  **Classification heads** output predictions for multiple tasks (pneumonia, tuberculosis, lung cancer).

This architecture efficiently combines the strengths of both DenseNet (feature reuse and deep representations) and Vision Transformers (modeling long-range dependencies), while keeping the model lightweight and computationally feasible for real-time or resource-constrained applications.
