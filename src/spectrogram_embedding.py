import os
import glob
import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

def main():
    input_dir = "data/spectrograms"
    embed_dir = "data/embeddings"
    
    # Ensure output directories exist
    os.makedirs(embed_dir, exist_ok=True)
    
    spectrogram_imgs = glob.glob(os.path.join(input_dir, "*.png"))
    
    if not spectrogram_imgs:
        print(f"No spectrogram images available in '{input_dir}' to embed.")
        print("Please run the generation script first.")
        return
        
    print(f"\nLoading facebook/dinov2-small model using Hugging Face Transformers...")
    try:
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        model = AutoModel.from_pretrained('facebook/dinov2-small')
        model.eval() # Set model to evaluation mode
        
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        model.to(device)
        print(f"Model loaded and mapped to device: {device}")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Generating DINOv2 vision embeddings for {len(spectrogram_imgs)} spectrogram images...")
    
    all_embeddings = []
    valid_image_paths = []
    
    with torch.no_grad():
        for i, img_path in enumerate(spectrogram_imgs, 1):
            try:
                print(f"[{i}/{len(spectrogram_imgs)}] Embedding {os.path.basename(img_path)}...")
                image = Image.open(img_path).convert('RGB')
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                outputs = model(**inputs)
                # Extract the CLS token embedding from the last hidden state
                # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
                # CLS token is at index 0.
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]
                
                all_embeddings.append(embedding)
                valid_image_paths.append(img_path)
                
            except Exception as e:
                print(f"    -> Failed to process {img_path}: {e}")
    
    if not all_embeddings:
        print("No embeddings were generated successfully.")
        return
        
    df = pd.DataFrame({
        "image_path": valid_image_paths,
        "embedding": all_embeddings
    })
    
    df['track_id'] = df['image_path'].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    
    out_parquet = os.path.join(embed_dir, "embedded_spectrograms.parquet")
    df.to_parquet(out_parquet)
    print(f"\nSuccessfully saved DINOv2 embeddings to {out_parquet}")

if __name__ == "__main__":
    main()
