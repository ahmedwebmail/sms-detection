# SMS Spam Detection Using Bernoulli Naive Bayes

A machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)**. The project covers data cleaning, exploratory data analysis (EDA), natural language preprocessing, feature extraction, model training, evaluation, and exporting trained artifacts for a Streamlit interface.

## Project Overview

The goal is to identify potentially unwanted SMS messages using a supervised binary classification model. Users can enter an SMS message in a Streamlit application and receive a **Spam** or **Not Spam** prediction.

**Labels:** `0 = Ham (Not Spam)` and `1 = Spam`.

## Technologies

- **Python**, **Google Colab**, **Jupyter Notebook**
- **pandas**, **NumPy** — data processing
- **Matplotlib**, **Seaborn** — visualization
- **NLTK** — tokenization, stopword removal, stemming
- **scikit-learn** — feature extraction, classification, evaluation
- **pickle** — model serialization
- **Streamlit** — optional web application interface

## Dataset

The notebook loads a CSV file named `spam.csv` with `cp1252` encoding. The original text and label columns are named `v2` and `v1`, respectively.

| Original column | Renamed column | Description |
| --- | --- | --- |
| `v1` | `target` | SMS category: Ham or Spam |
| `v2` | `text` | Original SMS message |

The notebook removes three additional unnamed columns, encodes the labels numerically, checks missing values, and removes duplicate records.

> The notebook references a Google Drive dataset path. Download or provide `spam.csv` yourself and update the path in the notebook before running it. Dataset source and license are not documented in the notebook.

## Workflow

```text
SMS Dataset (spam.csv)
        |
        v
Data Cleaning and Label Encoding
        |
        v
Exploratory Data Analysis
        |
        v
Text Preprocessing (NLTK)
        |
        v
CountVectorizer
        |
        v
Train/Test Split (80% / 20%)
        |
        v
Bernoulli Naive Bayes
        |
        v
Accuracy, Precision, Confusion Matrix
        |
        v
Export Model + Vectorizer (.pkl)
        |
        v
Streamlit SMS Prediction Interface
```

## Exploratory Data Analysis

The notebook investigates:

- Ham versus Spam class distribution using a pie chart.
- SMS character and token counts, with descriptive statistics by class.
- Message-length distributions using histograms.
- Correlations between numerical features using a heatmap.
- The 30 most frequent words in Ham and Spam messages using bar charts.

## Text Preprocessing

The notebook defines a function named **`tansform_text()`** (spelled this way in the source notebook), which:

1. Converts the message to lowercase.
2. Tokenizes text with NLTK.
3. Keeps alphanumeric tokens.
4. Removes English stopwords.
5. Applies Porter stemming.
6. Joins the processed tokens into a cleaned string.

Example (illustrative):

```text
Original: Congratulations! You have won a FREE prize.
Processed: congratul free prize
```

The cleaned messages are converted into numerical features using **CountVectorizer**.

## Model Training

The notebook defines three Naive Bayes classifiers:

- `BernoulliNB`

The current executed training section uses **BernoulliNB**; the Gaussian and Multinomial training/evaluation blocks are commented out.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred = bnb.predict(X_test)
```

## Evaluation

The notebook evaluates the Bernoulli Naive Bayes model using:

- **Accuracy:** proportion of correctly classified messages.
- **Precision:** proportion of predicted Spam messages that are actually Spam.
- **Confusion matrix:** counts of correct and incorrect predictions for both classes.

```python
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
```

**Results:** Add the actual metric values from your notebook run here. The uploaded notebook does not establish verified numeric results in this README.

| Metric | Result |
| --- | --- |
| Accuracy | To be added |
| Precision (Spam) | To be added |
| Confusion matrix | To be added |

## Saved Model Files

Two files are exported using `pickle`:

| File | Purpose |
| --- | --- |
| `bernoulli_nb.pkl` | Trained Bernoulli Naive Bayes classifier |
| `vectorizer.pkl` | Fitted CountVectorizer vocabulary and transformation |

Both files are required to transform and classify new messages consistently.

## Suggested Project Structure

```text
SMS-Spam-Detection/
├── Email-Spam-Classifier.ipynb
├── app.py                  # Streamlit interface, if included
├── bernoulli_nb.pkl        # Exported trained model
├── vectorizer.pkl          # Exported fitted vectorizer
├── requirements.txt
├── spam.csv                # Dataset, if redistribution is permitted
└── README.md
```

> `app.py`, the dataset, and model files are shown as suggested repository contents; only the notebook was supplied for this README.

## Installation and Usage

1. Clone the repository and open its folder:

   ```bash
   git clone https://github.com/YOUR_USERNAME/SMS-Spam-Detection.git
   cd SMS-Spam-Detection
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   ```

   On Windows, activate it with:

   ```powershell
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn nltk scikit-learn streamlit wordcloud notebook
   ```

4. Run `Email-Spam-Classifier.ipynb` in Jupyter or Google Colab, update the dataset path, and export the model and vectorizer.

5. If you have added the Streamlit `app.py` and saved `.pkl` files to the project folder, start the app:

   ```bash
   streamlit run app.py
   ```

   Open the local URL shown in the terminal (typically `http://localhost:8501`).

## Streamlit Prediction Flow

The intended interface allows users to enter an SMS message, submit it, and see **Spam SMS Detected** or **Not Spam SMS (Ham)**.

To keep inference consistent with training, the Streamlit app must apply the **same NLTK preprocessing** before calling the saved vectorizer and classifier:

```python
cleaned_sms = tansform_text(user_sms)
features = cv.transform([cleaned_sms])
prediction = model.predict(features)[0]

label = 'Spam SMS' if prediction == 1 else 'Not Spam SMS'
```

Example test inputs:

**Spam-like message:**

> Congratulations! You have won a free prize. Click here to claim now!

**Ham-like message:**

> Hi, are we still meeting tomorrow at 10 AM?

These are example inputs, not verified model outputs.

## Limitations and Improvements

- **Data leakage:** The notebook currently fits `CountVectorizer` on all messages **before** the train/test split. For reliable evaluation, split the raw texts first and fit the vectorizer only on training texts.
- **Metrics:** Record actual accuracy, precision, recall, F1-score, and confusion matrix from a clean evaluation run.
- **Model comparison:** Uncomment and evaluate GaussianNB and MultinomialNB to compare with BernoulliNB.
- **Preprocessing consistency:** Reuse the exact `tansform_text()` function during inference.
- **Notebook dependency:** The frequent-word analysis uses `Counter`; ensure `from collections import Counter` is imported.
- **Sentence counts:** The notebook's `num_sentence` calculation currently counts word tokens, not sentences. Use an appropriate sentence tokenizer if sentence-level analysis is required.
- **Reproducibility:** Pin compatible package versions and document the dataset source/license.
- **Security:** Load `.pkl` files only from trusted sources because pickle deserialization can execute code.

## Future Enhancements

- Compare multiple classifiers and tune hyperparameters.
- Add recall, F1-score, and a confusion-matrix heatmap.
- Package preprocessing and classification in one reproducible pipeline.
- Improve Streamlit styling and provide prediction feedback.
- Deploy the Streamlit app and add screenshots to this README.

## Example Message

- Spam SMS: Congratulations! You have won £1,000 in our lucky draw. Click here to claim your free prize now: http://claim-prize.example.com

- Not Spam SMS: Hi John, are we still meeting tomorrow at 10 AM? Let me know when you are free. Thanks!
## Disclaimer

This project is intended for learning and portfolio demonstration. Predictions are probabilistic model decisions and may be incorrect; it should not be treated as a production-grade messaging security filter without further validation.
