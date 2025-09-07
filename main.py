from scripts.load_data import load_data
from scripts.preprocess import preprocess_text
from scripts.visualize import plot_class_distribution, plot_top_words
from scripts.train_model import train_models
from scripts.evaluate import evaluate_model
from utils.helpers import print_divider

print_divider("Loading Data")
data = load_data("data/News.csv")

print_divider("Preprocessing Data")
data['text'] = preprocess_text(data['text'].values)

print_divider("Visualizations")
plot_class_distribution(data)
plot_top_words(data)

print_divider("Training Models")
log_reg, dtree, x_train, x_test, y_train, y_test = train_models(data)

print_divider("Evaluating Logistic Regression")
evaluate_model(log_reg, x_train, x_test, y_train, y_test, "Logistic Regression")

print_divider("Evaluating Decision Tree")
evaluate_model(dtree, x_train, x_test, y_train, y_test, "Decision Tree")
