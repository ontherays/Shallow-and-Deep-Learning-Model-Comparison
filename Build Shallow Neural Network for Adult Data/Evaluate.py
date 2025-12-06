test_loss, test_accuracy = shallow_model.evaluate(X_test, y_test, verbose=0)
print("Shallow Model Test Accuracy:", test_accuracy)
