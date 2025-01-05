# Neural Machine Translation using RNNs

This project demonstrates the use of Recurrent Neural Networks (RNNs) for Neural Machine Translation. The implementation uses TensorFlow and Keras to build and train a sequence-to-sequence model that translates text from English to French.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Training](#training)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the necessary libraries and dependencies, run:

```bash
pip install tensorflow numpy matplotlib sklearn opencv-python seaborn tensorflow_datasets tensorflow_probability
```

Additionally, you will need to install Kaggle to download the dataset:

```bash
pip install kaggle
```

## Data Preparation

### Data Download

The dataset is downloaded from [manythings.org](https://www.manythings.org/anki/fra-eng.zip) and extracted:

```python
!wget https://www.manythings.org/anki/fra-eng.zip
!unzip "/content/fra-eng.zip" -d "/content/dataset/"
```

### Kaggle Dataset

A larger dataset is downloaded from Kaggle:

```python
!pip install -q kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download -d dhruvildave/en-fr-translation-dataset
!unzip "/content/en-fr-translation-dataset.zip" -d "/content/dataset/"
```

### Data Processing

The data is processed and prepared for training by vectorizing the text sequences:

```python
text_dataset = tf.data.TextLineDataset("/content/dataset/fra.txt")
VOCAB_SIZE = 20000
ENGLISH_SEQUENCE_LENGTH = 64
FRENCH_SEQUENCE_LENGTH = 64
EMBEDDING_DIM = 300
BATCH_SIZE = 64

english_vectorize_layer = TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=ENGLISH_SEQUENCE_LENGTH
)

french_vectorize_layer = TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=FRENCH_SEQUENCE_LENGTH
)
```

## Modeling

The sequence-to-sequence model is built using Bidirectional GRU layers:

```python
input = Input(shape=(ENGLISH_SEQUENCE_LENGTH,), dtype="int64", name="input_1")
x = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(input)
encoded_input = Bidirectional(GRU(NUM_UNITS))(x)

shifted_target = Input(shape=(FRENCH_SEQUENCE_LENGTH,), dtype="int64", name="input_2")
x = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(shifted_target)
x = GRU(NUM_UNITS*2, return_sequences=True)(x, initial_state=encoded_input)

x = Dropout(0.5)(x)
target = Dense(VOCAB_SIZE, activation="softmax")(x)
seq2seq_gru = Model([input, shifted_target], target)
seq2seq_gru.summary()
```

## Training

The model is compiled and trained using the prepared dataset:

```python
seq2seq_gru.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(5e-4),
    metrics=['accuracy']
)

history = seq2seq_gru.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3,
    callbacks=[model_checkpoint_callback]
)
```

## Evaluation

The model is evaluated on the validation dataset:

```python
seq2seq_gru.evaluate(val_dataset)
```

## Testing

The model can be tested by feeding it new sentences and observing the translations.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
