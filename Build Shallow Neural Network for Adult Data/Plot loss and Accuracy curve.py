import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.plot(history_shallow.history['accuracy'])
plt.plot(history_shallow.history['val_accuracy'])
plt.title('Shallow Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'val'])
plt.show()

# Plot training & validation loss
plt.plot(history_shallow.history['loss'])
plt.plot(history_shallow.history['val_loss'])
plt.title('Shallow Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'val'])
plt.show()
