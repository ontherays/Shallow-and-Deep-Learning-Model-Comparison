history_deep = deep_model.fit(
    X_train, y_train_cat,
    validation_split=0.1,
    epochs=20,
    batch_size=128,
    verbose=1
)
