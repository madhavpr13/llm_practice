import datasets

emotions = datasets.load_dataset("emotion")
train_ds = emotions["train"]
print(f'features: {train_ds.features}')
print(f'column names: {train_ds.column_names}')

