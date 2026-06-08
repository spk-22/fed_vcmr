import torch
import torch.nn.functional as F
import numpy as np
import open_clip
import tensorflow as tf
import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

def check_alignment():
    print("--- Text Head Alignment Sanity Check ---")
    
    # 1. Load Reference PyTorch Model
    print("Loading MobileCLIP-S1 and best_model.pt...")
    model, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
    ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
    model.eval()
    
    # Load the specific head weights from the checkpoint
    # The head in export_tflite is a simple Linear layer
    class ProjectionHead(torch.nn.Module):
        def __init__(self, dim=512):
            super().__init__()
            self.linear = torch.nn.Linear(dim, dim, bias=False)
        def forward(self, x):
            return F.normalize(self.linear(x), dim=-1)

    pt_head = ProjectionHead()
    pt_head.load_state_dict(ckpt['text_head'])
    pt_head.eval()

    # 2. Load TFLite Model
    tflite_path = 'checkpoints/export/text_head_best_model.tflite'
    print(f"Loading TFLite head from {tflite_path}...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 3. Prepare Dummy Input (simulating backbone output)
    query = "a person walking on the beach"
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')
    tokens = tokenizer([query])
    
    with torch.no_grad():
        # Get backbone output (before head)
        # MobileCLIP-S1 text encoder structure: transformer -> ln_final -> text_projection
        # In our export script, we exported 'text_head' which takes the backbone output.
        backbone_output = model.encode_text(tokens)
        # Wait, if encode_text already includes the projection, we need to be careful.
        # OpenCLIP's encode_text for MobileCLIP:
        # x = self.token_embedding(text)
        # x = self.transformer(x)
        # x = self.ln_final(x)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        
        # Actually, best_model.pt 'text_head' was trained AS the projection.
        # Let's see how it was used in training/inference.
        # For now, let's just use a random vector to check if the weights/normalization match.
        test_input = torch.randn(1, 512)
        
        pt_output = pt_head(test_input).numpy()
        
        interpreter.set_tensor(input_details[0]['index'], test_input.numpy())
        interpreter.invoke()
        tf_output = interpreter.get_tensor(output_details[0]['index'])

    # 4. Compute Similarity
    cos_sim = np.dot(pt_output.flatten(), tf_output.flatten()) / (np.linalg.norm(pt_output) * np.linalg.norm(tf_output))
    
    print(f"\nResults for random vector:")
    print(f"PT Output norm: {np.linalg.norm(pt_output):.6f}")
    print(f"TF Output norm: {np.linalg.norm(tf_output):.6f}")
    print(f"Cosine Similarity: {cos_sim:.6f}")
    
    if cos_sim > 0.999:
        print("✅ SUCCESS: TFLite head matches PyTorch head weights and normalization.")
    else:
        print("❌ FAILURE: Discrepancy detected between heads.")
        # Check if one is normalized and the other isn't
        raw_pt = pt_head.linear(test_input).detach().numpy()
        raw_sim = np.dot(raw_pt.flatten(), tf_output.flatten()) / (np.linalg.norm(raw_pt) * np.linalg.norm(tf_output))
        print(f"Similarity (Raw PT vs TF): {raw_sim:.6f}")

if __name__ == "__main__":
    check_alignment()
