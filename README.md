### **Facial Emotion Recognition & Model Interpretability**

The repository is committed to the interpretability of machine learning models for emotion recognition in faces. The study explores sophisticated techniques such as LIME and SHAP in model outputs interpretation, offering transparency and trust in AI-based classification.

#### **Key Topicsady Covered:**
- **Interpretability in Classification**: How SVM and KNN models decide.
- **Emotion Recognition**: Evaluating different machine learning approaches for emotion classification from facial expressions.
- **Feature Engineering & Data Augmentation**: Improving the performance of models using maximized data sets.
- **Explainability Methods**: Applying post-hoc interpretability techniques (LIME, SHAP) in order to reason about black-box models.
- **Performance Analysis**: Evaluating interpretability v. classification accuracy trade-offs.

This research aims to enhance accuracy with responsible and transparent AI application in delicate fields like healthcare and finance.

#### **Summary results**
Best performance obtained by classifier SVM with kernel rbf using HOG as extractor with **88%** accuracy.
Using Lime 10 times per images, the regions used by the classifier are the following ones :

![image](https://github.com/user-attachments/assets/0456cb40-9c45-4123-a17d-3928c82ab0b5)

#### **Conclusion :**
The upper part of the face is crucial for recognizing facial expressions and emotions.
In this project, I used simple classifiers along with LIME to explain how the classifier works. A potentially more efficient approach would be to use SHAP for global explanations and explore the use of more complex classifiers

#### **Repository Structure:**

- /report → Full research report, including state of the art, methodology, results, and analysis.
- /dataset → Kaggle dataset ([CK+ Dataset](https://www.kaggle.com/datasets/davilsena/ckdataset)).
- /code → Machine learning models, training scripts, and explainability tools (LIME, SHAP).

### **Author**
[Emma Falkiewitz]
