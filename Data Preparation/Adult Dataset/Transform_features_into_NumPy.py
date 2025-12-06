# Fit only on training data
preprocessor.fit(X_train)

# Transform sets
X_train_nn = preprocessor.transform(X_train).toarray()
X_val_nn = preprocessor.transform(X_val).toarray()
X_test_nn = preprocessor.transform(X_test).toarray()

X_train_nn.shape, X_val_nn.shape, X_test_nn.shape
