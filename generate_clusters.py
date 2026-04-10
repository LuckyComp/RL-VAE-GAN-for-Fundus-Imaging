import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from models.varautoencoder import Encoder

def extract_latent_vectors(encoder, image_dir, device, image_size=256):
    """Passes all images through the frozen encoder to get their 'mu' vectors."""
    print(f"Extracting latent vectors from {image_dir}...")
    
    transforms = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image_paths = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    latent_vectors = []
    filenames = []

    with torch.no_grad():
        for filename in image_paths:
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transforms(img).unsqueeze(0).to(device) # Add batch dimension

            # Extract mu from the encoder
            mu, _ = encoder(img_tensor)
            
            latent_vectors.append(mu.cpu().numpy().squeeze())
            filenames.append(filename)

    return np.array(latent_vectors), filenames

if __name__ == "__main__":
    load_dotenv()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ENCODER_PATH = os.getenv("ENCODER_PATH", "./models/saved/encoder_pretrained.pth")
    DATASET_PATH = os.getenv("DATASET_PATH")
    TRAIN_DIR = os.path.join(DATASET_PATH, "train")
    NUM_CLUSTERS = 5
    OUTPUT_FILE = "cluster_labels.json"

    # 1. Load the frozen Encoder
    encoder = Encoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE, weights_only=True))
    encoder.eval()
    print("Encoder loaded successfully.")

    # 2. Extract Latent Vectors
    latent_vectors, filenames = extract_latent_vectors(encoder, TRAIN_DIR, DEVICE)
    print(f"Extracted {len(latent_vectors)} vectors of shape {latent_vectors.shape[1]}")

    # 3. Run K-Means Clustering
    print(f"Running K-Means clustering for {NUM_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(latent_vectors)

    # 4. Save the mapping to a JSON file
    mapping = {filenames[i]: int(cluster_ids[i]) for i in range(len(filenames))}
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(mapping, f, indent=4)

    print(f"Success! Saved {len(mapping)} labels to {OUTPUT_FILE}")
    
    # Optional: Print a quick summary of how many images are in each cluster
    unique, counts = np.unique(cluster_ids, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Cluster {u}: {c} images")
