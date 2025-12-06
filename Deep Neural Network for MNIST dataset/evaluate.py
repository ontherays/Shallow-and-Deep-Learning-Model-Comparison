test_loss, test_acc = deep_model.evaluate(X_test, y_test_cat, verbose=0)
print("Deep Model Test Accuracy:", test_acc)
