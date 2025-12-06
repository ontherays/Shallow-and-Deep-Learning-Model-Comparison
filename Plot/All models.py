import matplotlib.pyplot as plt

# --- SHALLOW ACCURACY ---
plt.figure()
plt.plot(history_shallow.history['accuracy'])
plt.plot(history_shallow.history['val_accuracy'])
plt.title('Shallow Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# --- SHALLOW LOSS ---
plt.figure()
plt.plot(history_shallow.history['loss'])
plt.plot(history_shallow.history['val_loss'])
plt.title('Shallow Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

# --- DEEP ACCURACY ---
plt.figure()
plt.plot(history_deep.history['accuracy'])
plt.plot(history_deep.history['val_accuracy'])
plt.title('Deep Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# --- DEEP LOSS ---
plt.figure()
plt.plot(history_deep.history['loss'])
plt.plot(history_deep.history['val_loss'])
plt.title('Deep Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()
