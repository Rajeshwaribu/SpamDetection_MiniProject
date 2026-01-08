A machine learning-based SMS/Email spam classifier built with Streamlit and deployed on Heroku. The project uses Natural Language Processing (NLP) techniques and Naive Bayes classification to accurately identify spam messages.

## Project Overview

This mini project demonstrates end-to-end machine learning workflow from data exploration to model deployment:
- **Dataset**: SMS Spam Collection with 5,572 messages
- **Accuracy**: 97.20%
- **Algorithm**: Multinomial Naive Bayes with TF-IDF vectorization

## Dataset

- **Source**: SMS Spam Collection Dataset
- **Total Messages**: 5,572 (5,169 after removing 403 duplicates)
- **Distribution**:
  - Ham (Legitimate): 87.37% (4,516 messages)
  - Spam: 12.63% (653 messages)
- **File**: `spam.csv`

### Class Imbalance
The dataset is imbalanced with significantly more ham messages than spam. The model handles this through appropriate algorithm selection (Naive Bayes performs well on imbalanced text data).

## Project Structure

```
SpamDetection_MiniProject/
├── app.py                          
├── sms-spam-detection.ipynb        
├── model.pkl                       
├── vectorizer.pkl                  
├── spam.csv                       
├── requirements.txt                
├── Procfile                          
├── setup.sh                        
├── nltk.txt                        
└── .gitignore                   
```

## Installation & Setup

### Prerequisites
- Python 3.7+
- pip or conda package manager

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SpamDetection_MiniProject
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (required for text preprocessing)
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the app**
   Open your browser and navigate to `http://localhost:8501`

## Usage

1. Launch the Streamlit app
2. Enter your SMS/Email message in the text area
3. Click the "Predict" button
4. The app will classify the message as either **Spam** or **Not Spam**

### Example Messages

**Spam Example:**
```
WINNER!! You have been selected to receive a $1000 prize.
Call now at 555-0123 to claim your reward!
```

**Ham Example:**
```
Hey, are we still meeting for lunch at 1pm today? Let me know!
```

## Model Development Process

### 1. Data Cleaning
- Removed 403 duplicate messages
- Dropped unnecessary columns (Unnamed: 2, 3, 4)
- Encoded target labels (ham=0, spam=1)

### 2. Exploratory Data Analysis (EDA)
- **Character Analysis**: Spam messages are longer (avg 137 chars) vs ham (avg 70 chars)
- **Word Count**: Spam has more words (avg 27) vs ham (avg 17)
- **Sentence Count**: Spam has more sentences (avg 3) vs ham (avg 2)
- **Word Clouds**: Visualized most common words in spam vs ham messages

### 3. Text Preprocessing Pipeline
```
1. Lowercase conversion
2. Tokenization (word_tokenize)
3. Remove special characters (keep alphanumeric only)
4. Remove stop words & punctuation
5. Porter Stemming (reduce words to root form)
```

**Example Transformation:**
```
Input:  "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight"
Output: "gon na home soon want talk stuff anymor tonight"
```

### 4. Feature Engineering
- **TF-IDF Vectorization**: max_features=3000
- Converts text into numerical vectors representing term importance
- Alternative features tested: character count, word count, sentence count

### 5. Model Selection & Evaluation

#### Models Tested
| Algorithm | Accuracy | Precision | Notes |
|-----------|----------|-----------|-------|
| **Multinomial Naive Bayes** | **97.20%** | **100%** | Best - Selected for deployment |
| Extra Trees Classifier | 97.68% | 99.15% | High performance but more complex |
| Random Forest | 97.49% | 98.28% | Excellent but slower |
| SVC (Sigmoid Kernel) | 97.29% | 97.41% | Good but slower training |
| XGBoost | 97.00% | 94.21% | Good but overkill for this problem |
| AdaBoost | 97.20% | 95.04% | Comparable to MNB |
| Logistic Regression | 96.13% | 97.12% | Simple and effective |
| Gradient Boosting | 94.87% | 92.93% | Slower with marginal gains |
| Bagging Classifier | 96.81% | 86.15% | Lower precision |
| Decision Tree | 94.39% | 83.81% | Overfitting risk |
| K-Nearest Neighbors | 92.84% | 77.12% | Poor performance on text |

#### Why Multinomial Naive Bayes?
- **100% Precision**: No false positives (crucial - don't want to block legitimate messages)
- **97.20% Accuracy**: Excellent overall performance
- **Fast Training & Inference**: Real-time predictions
- **Low Memory Footprint**: Small model size (model.pkl is only 96KB)
- **Proven for Text Classification**: Industry standard for spam detection

## Model Performance Metrics

### Confusion Matrix (Test Set)
```
                Predicted
              Ham    Spam
Actual Ham    896      0     ← Perfect! No legitimate messages marked as spam
     Spam      29    109    ← 79% spam detection rate
```

## Technical Stack

### Machine Learning & Data Science
- **scikit-learn**: Model training, evaluation, and vectorization
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **nltk**: Natural language processing (tokenization, stemming, stopwords)
- **XGBoost**: Gradient boosting framework (tested)

### Visualization
- **matplotlib**: Plotting and visualizations
- **seaborn**: Statistical data visualization
- **wordcloud**: Word cloud generation for EDA

### Web Application
- **Streamlit**: Interactive web interface for model deployment

### Deployment
- **Heroku**: Cloud platform for application hosting
- **pickle**: Model serialization

## Dependencies

```txt
streamlit
nltk
scikit-learn
```

Additional packages used in notebook:
- pandas
- numpy
- matplotlib
- seaborn
- wordcloud
- xgboost


## Learning Outcomes

This project demonstrates:
1. **End-to-End ML Pipeline**: From raw data to deployed application
2. **Text Classification**: NLP techniques for real-world problem
3. **Model Selection**: Comparing 10+ algorithms and choosing optimal solution
4. **Handling Imbalanced Data**: Strategies for skewed class distributions
5. **Feature Engineering**: TF-IDF vectorization for text data
6. **Web Deployment**: Creating interactive ML applications with Streamlit
7. **Cloud Deployment**: Hosting on Heroku with proper configuration

