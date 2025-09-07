from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(model, x_train, x_test, y_train, y_test, name="Model"):
    print(f"{name} Training Accuracy:", accuracy_score(y_train, model.predict(x_train)))
    print(f"{name} Testing Accuracy:", accuracy_score(y_test, model.predict(x_test)))

    cm = confusion_matrix(y_test, model.predict(x_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.show()
