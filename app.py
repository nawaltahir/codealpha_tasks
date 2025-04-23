from huggingface_hub import hf_hub_download
import gradio as gr
import joblib
import json
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
import numpy as np

# Load files from the model repo
repo_id = "Nawal20/Essay"

ridge_path = hf_hub_download(repo_id=repo_id, filename="ridge_model.pkl")
lgb_path = hf_hub_download(repo_id=repo_id, filename="lightgbm_model.pkl")
encoder_path = hf_hub_download(repo_id=repo_id, filename="scaler_encoder.pkl")
metadata_path = hf_hub_download(repo_id=repo_id, filename="metadata_columns.json")

# Load the models and encoder
ridge = joblib.load(ridge_path)
lgb_model = joblib.load(lgb_path)
encoder = joblib.load(encoder_path)

with open(metadata_path, "r") as f:
    metadata_columns = json.load(f)

# Load SBERT model
sbert = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")

def predict_score(essay_text, gender, race_ethnicity, assignment, prompt_name, disadvantaged, disability, ell_status):
    # Encode the essay
    essay_embedding = sbert.encode([essay_text])  # shape (1, 768)

    # Create metadata dict from input
    metadata_input = {
        "gender": gender,
        "race_ethnicity": race_ethnicity,
        "assignment": assignment,
        "prompt_name": prompt_name,
        "economically_disadvantaged": disadvantaged,
        "student_disability_status": disability,
        "ell_status": ell_status
    }

    # Create input array based on column order
    metadata_values = [metadata_input[col] for col in metadata_columns]
    metadata_array = encoder.transform([metadata_values])  # shape (1, N)

    # Combine essay + metadata
    full_input = np.hstack([essay_embedding.reshape(1, -1), metadata_array.toarray()])

    # Predict scores
    ridge_score = ridge.predict(full_input)[0]
    lgb_score = lgb_model.predict(full_input)[0]
    final_score = round((ridge_score + lgb_score) / 2, 2)

    return final_score

# Gradio UI
iface = gr.Interface(
    fn=predict_score,
    inputs=[
        gr.Textbox(label="Essay Text", lines=10, placeholder="Paste your essay here..."),
        gr.Dropdown(["Male", "Female", "Other"], label="Gender"),
        gr.Dropdown(["Asian", "Black", "Hispanic", "White", "Other"], label="Race/Ethnicity"),
        gr.Dropdown(["Informative", "Argumentative", "Narrative"], label="Assignment"),
        gr.Dropdown(["Education Benefits", "Technology Impact", "Climate Change"], label="Prompt Name"),
        gr.Dropdown(["Yes", "No"], label="Economically Disadvantaged"),
        gr.Dropdown(["None", "Learning", "Physical", "Other"], label="Student has Disability"),
        gr.Dropdown(["Yes", "No"], label="ELL Status"),
    ],
    outputs=gr.Number(label="Predicted Essay Score"),
    title="📘 Automated Essay Scoring App"
)

iface.launch()
