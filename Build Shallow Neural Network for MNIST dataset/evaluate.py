test_loss, test_acc = shallow_model.evaluate(X_test, y_test_cat, verbose=0)
print("Shallow Model Test Accuracy:", test_acc)
