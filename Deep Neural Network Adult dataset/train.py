test_loss_d, test_accuracy_d = deep_model.evaluate(X_test, y_test, verbose=0)
print("Deep Model Test Accuracy:", test_accuracy_d)
