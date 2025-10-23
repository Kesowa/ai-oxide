import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import torchvision.transforms as T
import torchvision
import torch
import os
import json


# === Configuration ===
onnx_model_path = "./retinanet-9.onnx"
image_path = "./test4.png"
rust_output_path = "rust_output.json"
output_image_path = "comparison_visualization.jpg"


# === Preprocessing ===
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((480, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    img = transform(image)
    return img.unsqueeze(0).numpy()


# === Simplified RetinaNet postprocessing ===
def detection_postprocess(image_tensor, cls_heads, box_heads, score_thresh=0.5, nms_thresh=0.5, top_k=100):
    """Lightweight decoding + NMS version for ONNX RetinaNet outputs"""
    anchors = []
    device = cls_heads[0].device
    batch_boxes, batch_scores, batch_labels = [], [], []

    strides = [8, 16, 32, 64, 128]
    ratios = [1.0, 2.0, 0.5]
    scales = [4 * 2 ** (i / 3) for i in range(3)]

    def generate_anchors(stride, ratios, scales):
        anchors = []
        for scale in scales:
            for ratio in ratios:
                w = stride * scale * (ratio ** 0.5)
                h = stride * scale / (ratio ** 0.5)
                anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(anchors, device=device)

    for i, (cls_head, box_head) in enumerate(zip(cls_heads, box_heads)):
        stride = strides[i]
        A = 9
        C = cls_head.shape[1] // A

        H, W = cls_head.shape[2:]
        anchors_level = generate_anchors(stride, ratios, scales)
        shift_x = torch.arange(0, W * stride, step=stride, device=device)
        shift_y = torch.arange(0, H * stride, step=stride, device=device)
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1).reshape(-1, 4)
        anchors_level = (anchors_level[None, :, :] + shifts[:, None, :]).reshape(-1, 4)

        cls_head = cls_head.permute(0, 2, 3, 1).reshape(-1, C)
        box_head = box_head.permute(0, 2, 3, 1).reshape(-1, 4)

        scores, labels = cls_head.sigmoid().max(dim=1)
        keep = scores > score_thresh
        scores, labels, box_head, anchors_level = scores[keep], labels[keep], box_head[keep], anchors_level[keep]

        boxes = torch.zeros_like(box_head)
        boxes[:, 0] = box_head[:, 0] * anchors_level[:, 2] + anchors_level[:, 0]
        boxes[:, 1] = box_head[:, 1] * anchors_level[:, 3] + anchors_level[:, 1]
        boxes[:, 2] = torch.exp(box_head[:, 2]) * anchors_level[:, 2]
        boxes[:, 3] = torch.exp(box_head[:, 3]) * anchors_level[:, 3]
        boxes[:, 2:] += boxes[:, :2]

        keep_idx = torchvision.ops.nms(boxes, scores, nms_thresh)
        keep_idx = keep_idx[:top_k]

        batch_boxes.append(boxes[keep_idx])
        batch_scores.append(scores[keep_idx])
        batch_labels.append(labels[keep_idx])

    boxes = torch.cat(batch_boxes)
    scores = torch.cat(batch_scores)
    labels = torch.cat(batch_labels)

    # === Keep only the most likely detection ===
    if len(scores) > 0:
        max_idx = torch.argmax(scores)
        boxes = boxes[max_idx].unsqueeze(0)
        scores = scores[max_idx].unsqueeze(0)
        labels = labels[max_idx].unsqueeze(0)
    
    return scores, boxes, labels



# === Run ONNX RetinaNet ===
def run_onnx_inference(model_path, image_path):
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    img = preprocess_image(image_path)
    print("Running ONNX inference...")
    outputs = session.run(None, {input_name: img})

    cls_heads = [torch.tensor(o) for o in outputs[:5]]
    box_heads = [torch.tensor(o) for o in outputs[5:]]

    scores, boxes, labels = detection_postprocess(torch.tensor(img), cls_heads, box_heads)
    return {"boxes": boxes.tolist(), "scores": scores.tolist(), "labels": labels.tolist()}


# === Load Rust JSON output ===
def load_rust_output(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


# === Draw boxes on image ===
def draw_boxes(image, boxes, color, width=2):
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(image)
    for box in boxes:
        if len(box) >= 4:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return image


# === Create side-by-side comparison ===
def create_comparison_image(image_path, python_boxes, rust_boxes, output_path):
    # Load and resize original image
    original_img = Image.open(image_path).convert("RGB")
    original_img.thumbnail((640, 480), Image.Resampling.LANCZOS)
    
    # Create two copies
    python_img = original_img.copy()
    rust_img = original_img.copy()
    
    # Draw boxes
    python_img = draw_boxes(python_img, python_boxes, color="lime", width=3)
    rust_img = draw_boxes(rust_img, rust_boxes, color="red", width=3)
    
    # Create side-by-side image
    total_width = python_img.width + rust_img.width + 30
    max_height = max(python_img.height, rust_img.height) + 60
    
    combined_img = Image.new("RGB", (total_width, max_height), color="white")
    
    # Paste images
    combined_img.paste(python_img, (10, 40))
    combined_img.paste(rust_img, (python_img.width + 20, 40))
    
    # Add labels
    draw = ImageDraw.Draw(combined_img)
    draw.text((python_img.width // 2 - 30, 10), "Python ONNX", fill="lime")
    draw.text((python_img.width + rust_img.width // 2 - 20, 10), "Rust Results", fill="red")
    
    combined_img.save(output_path)
    print(f"✅ Comparison image saved to {output_path}")


# === Main ===
if __name__ == "__main__":
    if not os.path.exists(onnx_model_path):
        print("Error: ONNX model not found.")
        exit(1)
    if not os.path.exists(image_path):
        print("Error: test image not found.")
        exit(1)
    if not os.path.exists(rust_output_path):
        print("Error: Rust output JSON not found.")
        exit(1)

    # Get Python results
    python_output = run_onnx_inference(onnx_model_path, image_path)
    python_boxes = python_output["boxes"]
    
    # Get Rust results
    rust_output = load_rust_output(rust_output_path)

    # === Keep only the most likely Rust detection ===
    if len(rust_output) > 0:
        best_rust = max(rust_output, key=lambda b: b.get("score", 0))
        rust_boxes = [best_rust["bounds"]]
    else:
        rust_boxes = []

    
    print(f"Python detected {len(python_boxes)} boxes")
    print(f"Rust detected {len(rust_boxes)} boxes")
    
    # Create comparison visualization
    create_comparison_image(image_path, python_boxes, rust_boxes, output_image_path)