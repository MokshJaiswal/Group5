import numpy as np
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100

def explain_prediction(text, model, tokenizer, label_encoder):
    class_names = label_encoder

    def predict_proba(texts):
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=MAX_LEN)
        return model.predict(padded)

    explainer = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        num_features=5
    )
    return dict(explanation.as_list())
