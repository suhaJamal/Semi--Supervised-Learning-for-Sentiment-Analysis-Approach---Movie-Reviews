## Semi-Supervised Learning for Sentiment Analysis on Movie Reviews

### Project Overview
This project applies semi-supervised learning techniques to perform sentiment analysis on movie reviews from the IMDB dataset. It explores the effectiveness of different classification models to determine sentiments expressed in movie reviews as either positive or negative.

### Live Demo
Check out the interactive web application to see the model in action: [Sentiment Analysis Web App](http://34.125.92.133:8501/)

### Dataset Link:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

### Objectives
- To apply semi-supervised learning methods for sentiment analysis within the Natural Language Processing (NLP) framework.
- To compare the performance of various models such as Logistic Regression, Multinomial Naive Bayes, and XGBoost in classifying sentiments.
- To utilize a limited amount of labeled data alongside a larger set of unlabeled data to improve model performance.

### Technologies Used
- Python
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, NLTK, Scikit-Learn, XGBoost, Keras, TensorFlow
- Tools: Jupyter Notebook, Streamlit for web application

### Installation and Usage
1. Clone this repository.
2. Install the required Python libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn nltk scikit-learn xgboost keras tensorflow streamlit
   ```
3. Run the Jupyter Notebooks to view the analysis and modeling.
4. Launch the Streamlit web app locally or visit the provided link.

### Data Description
The dataset includes 50,000 movie reviews from the IMDB database, labeled as positive or negative. The data features include:
- **Review**: Text of the review.
- **Sentiment**: Label indicating the sentiment (positive/negative).

### Methodology
The project follows the CRISP-DM process model, including:
- Data Understanding and Preprocessing
- Model Building and Evaluation
- Comparison of different semi-supervised learning techniques.

### Results
- **Model Comparison**: Logistic Regression, Multinomial Naive Bayes, and XGBoost models were evaluated with metrics such as accuracy, precision, recall, and F1-score.
- **Best Model**: Logistic Regression achieved the highest accuracy, demonstrating the effectiveness of traditional models in a semi-supervised setting with textual data.

### Contributing
Feel free to fork this repository and propose changes by submitting a pull request. We're open to any contributions or suggestions to improve the analysis and outcomes.
