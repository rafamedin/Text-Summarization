import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Attention, TimeDistributed, Concatenate, Lambda
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Load and preprocess the dataset
df = pd.read_csv('news_articles.csv')  # Load your dataset
df.dropna(inplace=True)  # Drop rows with missing values

# Preprocess text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize text
    filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    return ' '.join(filtered_tokens)

df['clean_text'] = df['text'].apply(preprocess_text)

# Tokenize text
max_vocab_size = 10000
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(df['clean_text'])

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df['clean_text'])
max_seq_length = max(len(seq) for seq in sequences)

# Pad sequences to ensure uniform length
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['summary'], test_size=0.2, random_state=42)

# Define custom attention layer with coverage mechanism
class AttentionLayerWithCoverage(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayerWithCoverage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_a = self.add_weight(shape=(input_shape[0][-1], input_shape[1][-1]),
                                   initializer='random_normal',
                                   trainable=True)
        self.U_a = self.add_weight(shape=(input_shape[1][-1], input_shape[1][-1]),
                                   initializer='random_normal',
                                   trainable=True)
        self.v_a = self.add_weight(shape=(input_shape[1][-1], 1),
                                   initializer='random_normal',
                                   trainable=True)
        super(AttentionLayerWithCoverage, self).build(input_shape)

    def call(self, inputs):
        encoder_outputs, decoder_outputs, coverage_vector = inputs
        e = K.dot(K.tanh(K.dot(encoder_outputs, self.W_a) + K.dot(decoder_outputs, self.U_a)), self.v_a)
        attention_weights = K.softmax(e, axis=1)
        context_vector = attention_weights * encoder_outputs
        context_vector_sum = K.sum(context_vector, axis=1)
        coverage_vector += attention_weights
        return context_vector_sum, attention_weights, coverage_vector

# Build the text summarization model with Pointer Generator Network
embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1

# Encoder
encoder_inputs = Input(shape=(max_seq_length,))
encoder_embeddings = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_inputs)
encoder_lstm = Bidirectional(LSTM(128, return_sequences=True))
encoder_outputs = encoder_lstm(encoder_embeddings)

# Decoder
decoder_inputs = Input(shape=(max_seq_length,))
decoder_embeddings = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(128, return_sequences=True)
decoder_outputs = decoder_lstm(decoder_embeddings, initial_state=encoder_lstm(encoder_embeddings))

# Attention with Coverage Mechanism
initial_coverage = tf.zeros_like(encoder_outputs[:, :, 0])  # Initialize coverage vector
attention_layer = AttentionLayerWithCoverage()
context_vector, attention_weights, coverage_vector = attention_layer([encoder_outputs, decoder_outputs, initial_coverage])

# Concatenate context vector with decoder outputs
decoder_combined_context = Concatenate(axis=-1)([decoder_outputs, tf.expand_dims(context_vector, axis=1)])

# Generate Vocabulary Distribution
output_layer = TimeDistributed(Dense(vocab_size, activation='softmax'))
vocabulary_distribution = output_layer(decoder_combined_context)

# Pointer Generator Network
P_gen = Dense(1, activation='sigmoid')
p_gen = P_gen(decoder_combined_context)

# Compute final distribution
final_distribution = Lambda(lambda x: x[0] * x[1] + (1 - x[0]) * x[2])([p_gen, vocabulary_distribution, attention_weights])

# Define model inputs and outputs
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=final_distribution)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_train, X_train], y_train, epochs=10, batch_size=64, validation_data=([X_test, X_test], y_test))

# Evaluate the model
loss, accuracy = model.evaluate([X_test, X_test], y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Generate summaries for new text
def generate_summary(text):
    cleaned_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_seq_length, padding='post')
    encoder_output = encoder_lstm(embedding_layer(padded_sequence))
    decoder_input = tf.convert_to_tensor([[tokenizer.word_index['<start>']]], dtype=tf.float32)
    summary = []
    for _ in range(max_seq_length):
        decoder_output = decoder_lstm(embedding_layer(decoder_input), initial_state=encoder_output)
        context_vector = attention_layer([encoder_output, decoder_output])
        attention_output = tf.concat([context_vector, decoder_output], axis=-1)
        output_probs = output_layer(attention_output)
        predicted_token_idx = tf.argmax(output_probs, axis=-1).numpy()[0, 0]
        if predicted_token_idx == tokenizer.word_index['<end>']:
            break
        summary.append(predicted_token_idx)
        decoder_input = tf.convert_to_tensor([[predicted_token_idx]], dtype=tf.float32)
    return tokenizer.sequences_to_texts([summary])

# Example usage
text = "A new study suggests that regular exercise can improve mental health and reduce stress."
summary = generate_summary(text)
print("Summary:", summary)
