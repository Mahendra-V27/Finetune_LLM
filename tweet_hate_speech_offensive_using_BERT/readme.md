# README

## Overview

This project demonstrates the process of building a text classification model using BERT for the task of classifying hate speech, offensive language, and neutral content. The dataset used is based on the Hate Speech and Offensive Language dataset, and the implementation uses Hugging Face's `transformers` library and TensorFlow for training a BERT-based model.

The steps involved include loading and preprocessing the dataset, tokenizing the text data, training a BERT model, and evaluating its performance on the validation and test sets. Finally, the trained model is used to predict classes for new text inputs.

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- Hugging Face Transformers
- Pandas, Seaborn, Matplotlib, and other visualization libraries

### Installation

You can install the required packages using pip:

```bash
pip install datasets transformers tensorflow pandas matplotlib seaborn
```

## Steps

### 1. Import Libraries

The necessary libraries for dataset loading, tokenization, model loading, and training are imported at the beginning.

```python
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
```

### 2. Load Dataset

The dataset is loaded from a Parquet file, cleaned, and split into train, test, and validation sets.

```python
pandas_df = pd.read_parquet("hf://datasets/tdavidson/hate_speech_offensive/data/train-00000-of-00001.parquet")
pandas_df['tweet_cleaned'] = pandas_df['tweet'].str.replace('@[A-Za-z0-9]+\s?', '', regex=True)
```

### 3. Tokenize Dataset

The BERT tokenizer is used to tokenize the cleaned text data. The text is converted into tokens, token type IDs, and attention masks.

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(dataset):
    return tokenizer(dataset['tweet_cleaned'], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

### 4. Prepare Data for Training

The tokenized datasets (train, test, and validation) are further prepared for training in TensorFlow by removing unnecessary columns and batching the data.

```python
train_set_for_final_model = tf.data.Dataset.from_tensor_slices((train_features, train_set['class'])).shuffle(len(train_set)).batch(8)
val_set_for_final_model = tf.data.Dataset.from_tensor_slices((eval_features, tf_eval_dataset["class"])).batch(8)
test_set_for_final_model = tf.data.Dataset.from_tensor_slices((test_features, tf_test_dataset["class"])).batch(8)
```

### 5. Load and Compile the Model

The BERT-based model is loaded from Hugging Face and compiled with a categorical cross-entropy loss function and Adam optimizer.

```python
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)
```

### 6. Train the Model

The model is trained for 3 epochs using the training dataset, and performance is validated on the validation dataset.

```python
history = model.fit(train_set_for_final_model, validation_data=val_set_for_final_model, epochs=3)
```

### 7. Evaluate the Model

The model is evaluated on the test dataset, and the accuracy is printed.

```python
test_loss, test_acc = model.evaluate(test_set_for_final_model, verbose=2)
print('Test accuracy:', test_acc)
```

### 8. Make Predictions

Once trained, the model can be used to predict the class (Hate Speech, Offensive Language, or Neither) of new text inputs.

```python
preds = model(tokenizer(["Sample text here"], return_tensors="tf", padding=True, truncation=True))['logits']
class_preds = np.argmax(preds, axis=1)
```

## Visualizations

- Model Accuracy: Plot showing training and validation accuracy over the epochs.
- Model Loss: Plot showing training and validation loss over the epochs.

```python
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()
```

## Conclusion

This project demonstrates a complete pipeline for text classification using BERT, including data loading, tokenization, model training, evaluation, and prediction. You can extend this project further by experimenting with different models, datasets, or fine-tuning strategies.
