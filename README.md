# Fake News Detection Using Natural Language Processing

## Overview
The rapid spread of fake news poses significant challenges to public trust and decision-making. This project develops a machine learning-based system to detect fake news using Natural Language Processing (NLP) techniques. By leveraging both traditional machine learning and advanced deep learning approaches, the project aims to provide accurate and scalable solutions for combating misinformation.

---

## Objectives
- Create an automated system that classifies news articles as real or fake.
- Implement both baseline and advanced models for text classification.
- Evaluate model performance using rigorous metrics such as accuracy and F1-score.
- Provide insights into feature importance and areas for improvement.

---

## Dataset
The dataset used was sourced from [Kaggle](https://www.kaggle.com/datasets/subho117/fake-news-detection-using-machine-learning). It includes **44,919 labeled news articles** with the following features:
- **title**: Headline of the article.
- **text**: Main content of the article.
- **subject**: Topic or category (e.g., "News").
- **date**: Publication date.
- **class**: Target variable (`0` for fake news, `1` for real news).

---

## Methodology
### Preprocessing
To prepare the data for modeling, the following steps were applied:
1. **Text Cleaning**: Removed punctuation, special characters, and numbers.
2. **Tokenization**: Split text into individual words.
3. **Lowercasing**: Standardized text by converting it to lowercase.
4. **Stopword Removal**: Removed common but uninformative words.
5. **Lemmatization**: Reduced words to their root forms.
6. **TF-IDF Vectorization**: Used for traditional models to emphasize unique terms.
7. **Tokenization and Padding**: For deep learning models, converted text to sequences of integers and padded them for uniformity.
8. **Data Splitting**: Divided the dataset into 75% training and 25% testing subsets.

---

## Models and Performance
### Baseline Models
1. **Logistic Regression**:
   - **Training Accuracy**: 99.31%
   - **Testing Accuracy**: 99.41%
   - **F1-Score**: 99%

2. **Decision Tree**:
   - **Training Accuracy**: 95.87%
   - **Testing Accuracy**: 95.07%
   - **F1-Score**: 95%

3. **Random Forest**:
   - **Training Accuracy**: 92.75%
   - **Testing Accuracy**: 89.76%
   - **F1-Score**: 89.5%

4. **Naive Bayes**:
   - **Training Accuracy**: 94.98%
   - **Testing Accuracy**: 94.89%
   - **F1-Score**: 94.5%

### Advanced Models
1. **Recurrent Neural Network (RNN)**:
   - **Training Accuracy**: 95.77%
   - **Testing Accuracy**: 94.82%
   - **F1-Score**: 95.5%
   - Captured sequential dependencies in text effectively.

2. **DistilBERT**:
   - **Training Accuracy**: 99.99%
   - **Testing Accuracy**: 99.99%
   - **F1-Score**: 99.9%
   - Leveraged attention mechanisms for contextual understanding, achieving near-perfect results.

---

## Key Observations
1. **Baseline Models**:
   - Logistic Regression demonstrated strong generalization due to its simplicity and effective use of TF-IDF features.
   - Ensemble methods like Random Forest showed moderate performance, limited by simpler text-based features.

2. **Advanced Models**:
   - DistilBERT significantly outperformed all models, highlighting the power of transformer-based architectures.
   - RNNs effectively modeled sequential relationships, offering notable improvements over baseline methods.

3. **Reasons for Improvement**:
   - **Contextual Understanding**: DistilBERT’s attention mechanisms captured nuanced relationships in text.
   - **Sequential Dependencies**: RNNs leveraged the order of words to enhance predictions.
   - **Feature Engineering**: Advanced embeddings provided deeper semantic insights compared to TF-IDF.

---

## Evaluation Metrics
To assess model performance, the following metrics were used:
- **Accuracy**: Overall correctness of predictions.
- **Precision**: Proportion of true positive predictions out of all positive predictions.
- **Recall**: Proportion of true positive predictions out of all actual positives.
- **F1-Score**: Harmonic mean of precision and recall, crucial for imbalanced datasets.

---

## Future Work
- Integrate additional datasets to improve robustness across diverse news sources.
- Explore advanced architectures like **GPT** or hybrid models.
- Implement model interpretability tools to enhance transparency in decision-making.
- Deploy the system in real-world scenarios to evaluate its effectiveness in dynamic environments.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries and Frameworks**:
  - `pandas`, `numpy`: Data manipulation.
  - `scikit-learn`: Baseline machine learning models.
  - `TensorFlow`, `PyTorch`: Deep learning implementation.
  - `transformers` (Hugging Face): For DistilBERT model fine-tuning.
  - `matplotlib`, `seaborn`: Data visualization.

## Author
### Shiven Sharma

