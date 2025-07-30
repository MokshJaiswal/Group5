import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from positional_encoding import PositionalEncoding, TransformerBlock

MAX_LEN = 200

def load_model_and_tokenizer():
    model = load_model("transformer_model.h5", custom_objects={
        "PositionalEncoding": PositionalEncoding,
        "TransformerBlock": TransformerBlock
    })

    with open("tokenizer.json", "r") as f:
        tokenizer_json = f.read()  # ✅ Read as string
        tokenizer = tokenizer_from_json(tokenizer_json)


    with open("label_encoder.json", "r") as f:
        label_encoder = json.load(f)

    return model, tokenizer, label_encoder

def predict_sentiment(text, model, tokenizer, label_encoder):
    # Tokenization
    seq = tokenizer.texts_to_sequences([text])
    print("Tokenized sequence:", seq)

    # Padding
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    print("Padded shape:", padded.shape)

    # Prediction
    pred = model.predict(padded)[0]
    print("Raw prediction probabilities:", pred)

    # Final label
    label = label_encoder[np.argmax(pred)]
    return label

