# Spam Detection using Naive Bayes

This project is a spam detection system that uses the Naive Bayes algorithm to classify SMS messages as "spam" or "ham" (not spam). The dataset used contains a collection of SMS messages labeled as either spam or ham.

## Dataset

The dataset used for this project is `spam.csv`, which contains the following columns:
- `v1`: Label indicating whether the message is "spam" or "ham".
- `v2`: The SMS message content.

## Project Structure

- `Spam_Detection.ipynb`: The main Jupyter notebook containing the implementation of the spam detection system.
- `spam.csv`: The dataset used for training and testing the model.

## Installation

To run this project, you'll need to install the following dependencies:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn wordcloud
```

## Usage

1. **Load the dataset**: The dataset is loaded using Pandas, with special attention to encoding issues that may arise due to invalid characters.

2. **Data Preprocessing**: 
    - Convert the labels to binary values: "spam" to 1 and "ham" to 0.
    - Vectorize the SMS messages using either `TfidfVectorizer` or `CountVectorizer`.

3. **Model Training**:
    - Split the data into training and testing sets using `train_test_split`.
    - Train the Naive Bayes classifier (`MultinomialNB`) on the training data.

4. **Model Evaluation**:
    - Evaluate the model using metrics such as ROC AUC score, F1 score, and confusion matrix.

5. **Visualization**:
    - Generate visualizations, including word clouds for spam and ham messages, and plot confusion matrices for model evaluation.

## Example Code

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Preprocess and vectorize the data
# ...

# Train the model
# ...

# Evaluate the model
# ...
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
