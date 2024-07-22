from flask import Flask, render_template, request, send_from_directory
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input values from the form
        num_classes = int(request.form['num_classes'])
        class_names = [request.form[f'class_{i}_name'] for i in range(num_classes)]
        num_images = [int(request.form[f'class_{i}_images']) for i in range(num_classes)]
        accuracy_needed = float(request.form['accuracy'])
        epoch_count = int(request.form['epochs'])
        display_labels = class_names

        # Generate random data
        y_true = np.random.randint(0, num_classes, sum(num_images))
        y_pred = np.random.randint(0, num_classes, sum(num_images))

        # Simulate confusion matrix where only Class 1 has counts
        correct_indices = np.random.choice(np.arange(len(y_true)), size=int(len(y_true) * accuracy_needed), replace=False)
        y_pred[correct_indices] = y_true[correct_indices]

        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        # Plot Confusion Matrix
        plot_dir = os.path.join('static', 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
            xticklabels=display_labels, 
            yticklabels=display_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        cm_path = os.path.join(plot_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()

        # Plot Combined ROC Curve
        plt.figure()
        for i in range(num_classes):
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{display_labels[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Classes')
        plt.legend(loc="lower right")
        combined_roc_path = os.path.join(plot_dir, 'combined_roc_curve.png')
        plt.savefig(combined_roc_path)
        plt.close()

        # Plot Combined Precision-Recall Curve
        plt.figure()
        for i in range(num_classes):
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_binary)
            plt.plot(recall, precision, label=display_labels[i])
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for All Classes')
        plt.legend(loc="lower left")
        combined_pr_path = os.path.join(plot_dir, 'combined_precision_recall_curve.png')
        plt.savefig(combined_pr_path)
        plt.close()

       
        # Simulate Accuracy and Loss over Epochs
        epochs = np.arange(1, epoch_count + 1)
        accuracy_values = np.linspace(0.5, accuracy_needed, epoch_count) + np.random.normal(0, 0.02, epoch_count)
        loss_values = np.linspace(1.0, 0.1, epoch_count) + np.random.normal(0, 0.02, epoch_count)

        plt.figure()
        plt.plot(epochs, accuracy_values, label='Accuracy', marker='o')
        plt.plot(epochs, loss_values, label='Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Accuracy and Loss over Epochs')
        plt.legend(loc="upper right")
        acc_loss_path = os.path.join(plot_dir, 'accuracy_loss.png')
        plt.savefig(acc_loss_path)
        plt.close()

        return render_template('results.html',
                               cm=cm,
                               accuracy=accuracy,
                               precision=np.mean(precision),
                               recall=np.mean(recall),
                               f1=np.mean(f1),
                               combined_roc_path=combined_roc_path,
                               combined_pr_path=combined_pr_path,
                               acc_loss_path=acc_loss_path,
                               cm_path=cm_path)

    return render_template('index.html')


# Route to serve plot files
@app.route('/plots/<filename>')
def plot_file(filename):
    return send_from_directory(os.path.join('static', 'plots'), filename)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0")
