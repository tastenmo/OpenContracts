import os

from sentence_transformers import SentenceTransformer

# Directory to save the model (absolute path)
#cache_dir = "/models"

# Full path where the model will be saved
model_save_path = "/models"

# Download and save the sentence transformer model
model = SentenceTransformer(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
#    "all-MiniLM-L6-v2",
)

# Save the model to the desired path
model.save(model_save_path)

print(
    f"Sentence transformer model has been downloaded and saved to '{model_save_path}'."
)
