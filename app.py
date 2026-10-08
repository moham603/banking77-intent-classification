import tensorflow as tf
import joblib
import streamlit as st

from tensorflow.keras import layers


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="BANKING77",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# Custom Transformer Layers
# =========================================================

class TransformerBlock(layers.Layer):

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )

        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])

        self.layernorm1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.layernorm2 = layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def build(self, input_shape):

        self.attention.build(
            input_shape,
            input_shape
        )

        self.ffn.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, training=False):

        attention_output = self.attention(
            inputs,
            inputs
        )

        attention_output = self.dropout1(
            attention_output,
            training=training
        )

        out1 = self.layernorm1(
            inputs + attention_output
        )

        ffn_output = self.ffn(out1)

        ffn_output = self.dropout2(
            ffn_output,
            training=training
        )

        return self.layernorm2(
            out1 + ffn_output
        )


class PositionalEmbedding(layers.Layer):

    def __init__(
        self,
        sequence_length,
        vocab_size,
        embed_dim,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True
        )

        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length,
            output_dim=embed_dim
        )

    def build(self, input_shape):

        self.token_embeddings.build(input_shape)

        self.position_embeddings.build(
            (self.sequence_length,)
        )

        super().build(input_shape)

    def call(self, inputs):

        length = tf.shape(inputs)[-1]

        positions = tf.range(
            start=0,
            limit=length,
            delta=1
        )

        embedded_tokens = self.token_embeddings(
            inputs
        )

        embedded_positions = self.position_embeddings(
            positions
        )

        return embedded_tokens + embedded_positions


# =========================================================
# Load Model
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "final_transformer.keras",
        custom_objects={
            "TransformerBlock": TransformerBlock,
            "PositionalEmbedding": PositionalEmbedding
        }
    )


@st.cache_resource
def load_encoder():

    return joblib.load(
        "label_encoder.pkl"
    )


try:

    loaded_model = load_model()
    loaded_label_encoder = load_encoder()

except Exception as e:

    st.error("Unable to load the model files.")

    st.info(
        "Make sure these files exist in the same folder as app.py:\n\n"
        "• final_transformer.keras\n"
        "• label_encoder.pkl"
    )

    st.stop()


# =========================================================
# Prediction Function
# =========================================================

def predict_intent(text):

    input_text = tf.constant([text])

    prediction_probs = loaded_model.predict(
        input_text,
        verbose=0
    )

    predicted_class = prediction_probs.argmax(
        axis=1
    )[0]

    predicted_intent = loaded_label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = prediction_probs[0][
        predicted_class
    ]

    return predicted_intent, confidence, prediction_probs[0]


# =========================================================
# Styling
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0b1120;
    }

    [data-testid="stHeader"] {
        background-color: transparent;
    }

    [data-testid="stToolbar"] {
        visibility: hidden;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Header
# =========================================================

st.title("🤖 BANKING77")

st.subheader("Customer Intent Classifier")

st.caption(
    "Transformer-based NLP application for banking customer intent prediction."
)

st.divider()


# =========================================================
# Model Overview
# =========================================================

st.subheader("Model Overview")

st.caption(
    "Key information about the trained Transformer model."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model",
        "Transformer"
    )

with col2:
    st.metric(
        "Dataset",
        "BANKING77"
    )

with col3:
    st.metric(
        "Classes",
        "77"
    )

with col4:
    st.metric(
        "Task",
        "Text Classification"
    )


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Sequence Length",
        "50"
    )

with col2:
    st.metric(
        "Embedding",
        "128"
    )

with col3:
    st.metric(
        "Attention Heads",
        "8"
    )

with col4:
    st.metric(
        "Validation Accuracy",
        "88.87%"
    )


st.divider()


# =========================================================
# Customer Message
# =========================================================

st.subheader("💬 Customer Message")

st.caption(
    "Enter a banking-related message to predict the customer's intent."
)

user_text = st.text_area(
    "Message",
    placeholder=(
        "Example:\n"
        "I forgot my password"
    ),
    height=150,
    label_visibility="collapsed"
)


# =========================================================
# Buttons
# =========================================================

predict_col, clear_col, empty_col = st.columns(
    [1, 1, 2]
)

with predict_col:

    predict_button = st.button(
        "🔮 Predict Intent",
        use_container_width=True
    )

with clear_col:

    clear_button = st.button(
        "🗑️ Clear",
        use_container_width=True
    )


if clear_button:

    st.rerun()


# =========================================================
# Prediction
# =========================================================

if predict_button:

    if not user_text.strip():

        st.warning(
            "Please enter a customer message first."
        )

    else:

        with st.spinner("Analyzing customer message..."):

            try:

                intent, confidence, probabilities = predict_intent(
                    user_text
                )

                display_intent = intent.replace(
                    "_",
                    " "
                ).title()

                confidence_percent = confidence * 100

                st.divider()

                st.subheader("🎯 Prediction Result")

                result_col1, result_col2 = st.columns(
                    [2, 1]
                )

                with result_col1:

                    st.caption(
                        "PREDICTED INTENT"
                    )

                    st.success(
                        display_intent
                    )

                    st.caption(
                        f"Model class: `{intent}`"
                    )

                with result_col2:

                    st.metric(
                        "Confidence",
                        f"{confidence_percent:.2f}%"
                    )

                st.progress(
                    float(confidence)
                )

                if confidence >= 0.90:

                    st.success(
                        "High-confidence prediction"
                    )

                elif confidence >= 0.70:

                    st.info(
                        "Moderate-confidence prediction"
                    )

                else:

                    st.warning(
                        "Low-confidence prediction"
                    )

                # =================================================
                # Top 3 Predictions
                # =================================================

                st.subheader("📊 Top 3 Predictions")

                top_indices = probabilities.argsort()[
                    -3:
                ][::-1]

                for rank, index in enumerate(
                    top_indices,
                    start=1
                ):

                    class_name = (
                        loaded_label_encoder
                        .inverse_transform([index])[0]
                    )

                    readable_name = class_name.replace(
                        "_",
                        " "
                    ).title()

                    score = probabilities[index]

                    top_col1, top_col2 = st.columns(
                        [3, 1]
                    )

                    with top_col1:

                        st.write(
                            f"**{rank}. {readable_name}**"
                        )

                        st.caption(
                            f"`{class_name}`"
                        )

                    with top_col2:

                        st.write(
                            f"**{score:.2%}**"
                        )

                    st.progress(
                        float(score)
                    )


            except Exception as e:

                st.error(
                    "An error occurred while making the prediction."
                )

                st.exception(e)


# =========================================================
# Example Messages
# =========================================================

st.divider()

st.subheader("🧪 Try These Examples")

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:

    st.info(
        "💳 **Card Payment**\n\n"
        "My card was declined when I tried to pay."
    )

with example_col2:

    st.info(
        "🔐 **Password**\n\n"
        "I forgot my password."
    )

with example_col3:

    st.info(
        "💸 **Transfer**\n\n"
        "How long will my transfer take?"
    )


# =========================================================
# Model Performance
# =========================================================

st.divider()

st.subheader("📈 Model Performance")

perf_col1, perf_col2, perf_col3 = st.columns(3)

with perf_col1:

    st.metric(
        "Validation Accuracy",
        "88.87%"
    )

with perf_col2:

    st.metric(
        "Test Accuracy",
        "87.40%"
    )

with perf_col3:

    st.metric(
        "Test Macro F1",
        "87.43%"
    )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "BANKING77 Intent Classifier  •  "
    "Python  •  TensorFlow  •  Streamlit"
)