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
