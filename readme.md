# Fake News Detection Project 📰

This project is a machine learning application designed to classify news articles as either real or fake. It utilizes Natural Language Processing (NLP) techniques to preprocess text data and trains two different classification models—Logistic Regression and a Decision Tree—to perform the prediction.


## 📋 Table of Contents
* [Project Workflow](#-project-workflow)
* [Dataset](#-dataset)
* [Technologies Used](#-technologies-used)
* [Setup and Usage](#-setup-and-usage)
* [Results](#-results)
* [License](#-license)

## ⚙️ Project Workflow

The project follows these key steps:
1.  **Data Loading**- https://drive.google.com/file/d/1q5jpI5M1EA9x3YPrLupmiu3gffkmGlHj/view
2.  **Initial Exploration**: The `News.csv` dataset is loaded into a pandas DataFrame. Unnecessary columns (`title`, `subject`, `date`) are dropped.
3.  **Data Cleaning & Preprocessing**:
    * The dataset is shuffled to ensure randomness.
    * A custom function cleans the text of each news article by removing punctuation, converting all text to lowercase, and filtering out common English stopwords.
4.  **Data Visualization**: Key insights are visualized, including the distribution of real vs. fake news articles and a bar chart of the most frequently used words in the corpus.
5.  **Feature Extraction**: The cleaned text data is converted into a numerical format using the **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization technique.
6.  **Model Training**: The dataset is split into training (75%) and testing (25%) sets. Two models are trained on this data:
    * **Logistic Regression**
    * **Decision Tree Classifier**
7.  **Model Evaluation**: The performance of both models is evaluated on the training and testing sets using the **accuracy score**. A **confusion matrix** is also generated for the Decision Tree model to visualize its performance in detail.

## 💾 Dataset
The project uses the `News.csv` dataset, which contains news articles labeled as either real or fake. The primary columns used for this analysis are:
* `text`: The full text of the news article.
* `class`: The target label (e.g., 'REAL' or 'FAKE', 0 or 1).

## 🛠️ Technologies Used
* **Python 3.x**
* **Pandas**: For data manipulation and analysis.
* **NLTK (Natural Language Toolkit)**: For text preprocessing tasks like tokenization and stopword removal.
* **Scikit-learn**: For machine learning, including model implementation (Logistic Regression, Decision Tree), feature extraction (TfidfVectorizer), and evaluation metrics.
* **Matplotlib & Seaborn**: For data visualization.
* **TQDM**: For displaying smart progress bars during long processes.

## 🚀 Setup and Usage

Follow these steps to run the project locally.

**1. Clone the repository:**
```bash
git clone https://github.com/1Adarsh-git/fake-news-prediction.git
cd newsdetector
````

**2. Create and activate a virtual environment:**

  * This keeps your project dependencies isolated.

<!-- end list -->

```bash
# Create the environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

**3. Install dependencies:**

```bash

# Install dependencies
pip install -r requirements.txt
```

**4. Run the script:**

  * Execute the main Python file from your terminal.

<!-- end list -->

```bash
python3 main.py
```

  * The script will print the model accuracy scores and display the generated plots.

## 📈 Results

The script evaluates two models and prints their accuracy on the training and test data. The output will look similar to this:

  * **Logistic Regression Accuracy**:
      * Training Accuracy: `0.993`
      * Test Accuracy: `0.989`
  * **Decision Tree Accuracy**:
      * Training Accuracy: `0.999`
      * Test Accuracy: `0.995`

Additionally, a confusion matrix is displayed for the Decision Tree model, providing a clear view of its true positives, true negatives, false positives, and false negatives.

### 🤝 Contributing

This project is designed for learning and experimentation. Feel free to:
- 🔧 Modify the synthetic data generation logic
- 🤖 Try different ML algorithms  
- 📊 Add new visualizations
- 🎨 Enhance the Streamlit UI
- 📝 Improve documentation

## 📄 License

This project is for educational purposes. Feel free to use it in your portfolio, hackathons, or learning journey!

## 👨‍💻 Author

**Built as a learning project**

---
