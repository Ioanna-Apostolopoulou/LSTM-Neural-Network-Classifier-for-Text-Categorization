import os
import pickle
import matplotlib.pyplot as plt

HISTORY_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'history.pkl')

with open(HISTORY_PATH, 'rb') as file:
    data = pickle.load(file)

    # Creating the loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['loss'], label='Training Loss')
    plt.plot(data['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png', dpi=300)

    # Creating the accuracy plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axes[0].plot(data['category_level_1_sparse_categorical_accuracy'], label='Training Accuracy (Category 1)')
    axes[0].plot(data['val_category_level_1_sparse_categorical_accuracy'], label='Validation Accuracy (Category 1)')
    axes[0].set_title('Training and Validation Accuracy for Category 1')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(data['category_level_2_sparse_categorical_accuracy'], label='Training Accuracy (Category 2)')
    axes[1].plot(data['val_category_level_2_sparse_categorical_accuracy'], label='Validation Accuracy (Category 2)')
    axes[1].set_title('Training and Validation Accuracy for Category 2')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    plt.savefig('accuracy_plot.png', dpi=300)