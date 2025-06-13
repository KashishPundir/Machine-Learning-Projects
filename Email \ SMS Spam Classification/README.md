# Email & SMS Spam Detection using Machine Learning 📧
This project is all about automating email spam detection using Machine Learning. It showcases how we can turn messy email text into meaningful patterns and build a model that accurately separates spam from ham (non-spam) messages.

## 📌 **Key Highlights**  
✅ Text preprocessing with NLTK  
✅ Feature extraction using TF-IDF  
✅ Multiple ML models tested (Naive Bayes, Logistic Regression, Random Forest etc.)  
✅ Strong evaluation using accuracy, precision, recall & confusion matrix  

## 🗃️ **Dataset:**  
The dataset consists of labeled email messages.  
**Labels:**  
    ham = Not Spam  
    spam = Unwanted emails  

The data is loaded and explored using pandas.  

## 🔄 **Workflow:**  
1️⃣ Data Cleaning & Preprocessing  
🔸Lowercasing  
🔸Tokenization  
🔸Removing punctuation and stopwords  
🔸Stemming using PorterStemmer  

2️⃣ Feature Engineering  
TF-IDF Vectorization to convert text into numeric format  

3️⃣ Model Training  
Tested with:  
🔸 Naive Bayes  
🔸 Logistic Regression  
🔸 Random Forest etc....  

## 4️⃣ **Evaluation:**  
Accuracy, Precision, Recall, F1-Score, Confusion Matrix  

## 📊 **Results:**  
🔹High Accuracy & Precision  
🔹Effectively detects spam even in short, tricky messages  
🔹Balanced performance across all classes  

## 🧰 **Tools & Tech Stack**:  

🐍 Python :	Programming Language  
📦 Pandas :	Data Manipulation  
🔢 NumPy : Numerical Computing  
🧠 Scikit-learn	Machine : Learning Models  
🗣️ NLTK :	Natural Language Processing  
📊 Matplotlib / Seaborn :	Visualizations  

## 🛠️ **How to Run This Project:**  
1. Clone or download the repository

2. Install required libraries:
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn
```
3. Open and run the Jupyter notebook: **Email & SMS Spam Detection.ipynb**


💬 **Thank you for visiting this project! Feel free to explore, fork, or contribute.** 🙌

