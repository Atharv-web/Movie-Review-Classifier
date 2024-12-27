# Movie Review Classifier (MRC)

## Overview
The **Movie Review Classifier (MRC)** repository contains two implementations of sentiment analysis for movie reviews. The primary objective is to classify reviews as either **positive** or **negative** based on their textual content. This repository uses the widely recognized IMDB dataset for training and evaluation.

## Implementations

### 1. Convolutional Neural Network (CNN)
This implementation leverages deep learning techniques for sentiment analysis. Using a Convolutional Neural Network, the program processes textual data to extract meaningful patterns and classify reviews efficiently.

#### Features:
- Implements a CNN architecture for text classification.
- Includes preprocessing steps like tokenization and embedding layer usage.
- Optimized for handling large datasets such as the IMDB dataset.

#### Requirements:
- Python 3.x
- TensorFlow/Keras or PyTorch
- Supporting libraries: NumPy, Pandas, Matplotlib

### 2. Naive Bayes (NB)
This implementation utilizes the Naive Bayes classifier for sentiment analysis. It provides a simpler yet effective approach to text classification, suitable for smaller datasets or as a baseline comparison.

#### Features:
- Employs the Naive Bayes algorithm for text classification.
- Includes preprocessing steps like stopword removal and vectorization.
- Lightweight and easy to implement on most systems.

#### Requirements:
- Python 3.x
- Scikit-learn
- Supporting libraries: NumPy, Pandas

## Dataset
The repository uses the **IMDB dataset** for sentiment analysis. This dataset contains 50,000 movie reviews, equally divided into positive and negative sentiments. It is widely used for natural language processing (NLP) tasks.

## Getting Started
1. Clone the repository:
   ```bash
   git clone movie-review-classifier
   ```

2. Install the necessary dependencies for the chosen implementation.
   - For CNN:
     ```bash
     pip install tensorflow numpy pandas matplotlib
     ```
   - For NB:
     ```bash
     pip install scikit-learn numpy pandas
     ```

3. Download and prepare the IMDB dataset if not included in the repository.

4. Run the desired implementation:
   - CNN:
     ```bash
     python mrc-cnn.ipynb
     ```
   - NB:
     ```bash
     python mrc-nb.ipynb
     ```
