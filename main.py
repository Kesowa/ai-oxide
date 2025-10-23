from torchvision import transforms
import torch
import onnxruntime
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def flatten(inputs):
    return [
        [flatten(i) for i in inputs] if isinstance(inputs, (list, tuple)) else inputs
    ]


def update_flatten_list(inputs, res_list):
    for i in inputs:
        res_list.append(i) if not isinstance(i, (list, tuple)) else update_flatten_list(
            i, res_list
        )
    return res_list


def to_numpy(x):
    if type(x) is not np.ndarray:
        x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
    return x


preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def generate_anchors(stride, ratio_vals, scales_vals, angles_vals=None):
    "Generate anchors coordinates from scales/ratios"

    scales = torch.FloatTensor(scales_vals).repeat(len(ratio_vals), 1)
    scales = scales.transpose(0, 1).contiguous().view(-1, 1)
    ratios = torch.FloatTensor(ratio_vals * len(scales_vals))

    wh = torch.FloatTensor([stride]).repeat(len(ratios), 2)
    ws = torch.sqrt(wh[:, 0] * wh[:, 1] / ratios)
    dwh = torch.stack([ws, ws * ratios], dim=1)
    xy1 = 0.5 * (wh - dwh * scales)
    xy2 = 0.5 * (wh + dwh * scales)
    return torch.cat([xy1, xy2], dim=1)


def box2delta(boxes, anchors):
    "Convert boxes to deltas from anchors"

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
    boxes_wh = boxes[:, 2:] - boxes[:, :2] + 1
    boxes_ctr = boxes[:, :2] + 0.5 * boxes_wh

    return torch.cat(
        [(boxes_ctr - anchors_ctr) / anchors_wh, torch.log(boxes_wh / anchors_wh)], 1
    )


def delta2box(deltas, anchors, size, stride):
    "Convert deltas from anchors to boxes"

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + ctr
    pred_wh = torch.exp(deltas[:, 2:]) * anchors_wh

    m = torch.zeros([2], device=deltas.device, dtype=deltas.dtype)
    M = torch.tensor([size], device=deltas.device, dtype=deltas.dtype) * stride - 1
    clamp = lambda t: torch.max(m, torch.min(t, M))
    return torch.cat(
        [clamp(pred_ctr - 0.5 * pred_wh), clamp(pred_ctr + 0.5 * pred_wh - 1)], 1
    )


def decode(
    all_cls_head,
    all_box_head,
    stride=1,
    threshold=0.05,
    top_n=1000,
    anchors=None,
    rotated=False,
):
    "Box Decoding and Filtering"

    if rotated:
        anchors = anchors[0]
    num_boxes = 4 if not rotated else 6

    device = all_cls_head.device
    anchors = anchors.to(device).type(all_cls_head.type())
    num_anchors = anchors.size()[0] if anchors is not None else 1
    num_classes = all_cls_head.size()[1] // num_anchors
    height, width = all_cls_head.size()[-2:]

    batch_size = all_cls_head.size()[0]
    out_scores = torch.zeros((batch_size, top_n), device=device)
    out_boxes = torch.zeros((batch_size, top_n, num_boxes), device=device)
    out_classes = torch.zeros((batch_size, top_n), device=device)

    # Per item in batch
    for batch in range(batch_size):
        cls_head = all_cls_head[batch, :, :, :].contiguous().view(-1)
        box_head = all_box_head[batch, :, :, :].contiguous().view(-1, num_boxes)

        # Keep scores over threshold
        keep = (cls_head >= threshold).nonzero().view(-1)
        if keep.nelement() == 0:
            continue

        # Gather top elements
        scores = torch.index_select(cls_head, 0, keep)
        scores, topk_indices = torch.topk(scores, min(top_n, keep.size()[0]), dim=0)
        # map back to original flattened indices
        indices = torch.index_select(keep, 0, topk_indices).view(-1)  # LONG tensor

        # compute channel/anchor/xy using integer division
        # flattened index layout: idx = c*(H*W) + y*W + x
        c = indices // (height * width)  # channel index (c)
        a = c // num_classes  # anchor index
        classes = (c % num_classes).long()  # class index (as long for safety)

        x = indices % width
        y = (indices // width) % height

        # ensure integer tensors for indexing
        a = a.long()
        y = y.long()
        x = x.long()

        # type for classes to store in out_classes (float-compatible)
        classes_float = classes.type(all_cls_head.type())

        # Reshape box_head to [num_anchors, num_boxes, height, width] for indexing
        box_head = box_head.view(num_anchors, num_boxes, height, width)

        # Index boxes with Long tensors (advanced indexing)
        boxes = box_head[a, :, y, x]

        if anchors is not None:
            grid = (
                torch.stack([x, y, x, y], 1).type(all_cls_head.type()) * stride
                + anchors[a, :]
            )
            boxes = delta2box(boxes, grid, [width, height], stride)

        out_scores[batch, : scores.size()[0]] = scores
        out_boxes[batch, : boxes.size()[0], :] = boxes
        out_classes[batch, : classes_float.size()[0]] = classes_float

    return out_scores, out_boxes, out_classes


def nms(all_scores, all_boxes, all_classes, nms=0.5, ndetections=100):
    "Non Maximum Suppression"

    device = all_scores.device
    batch_size = all_scores.size()[0]
    out_scores = torch.zeros((batch_size, ndetections), device=device)
    out_boxes = torch.zeros((batch_size, ndetections, 4), device=device)
    out_classes = torch.zeros((batch_size, ndetections), device=device)

    # Per item in batch
    for batch in range(batch_size):
        # Discard null scores
        keep = (all_scores[batch, :].view(-1) > 0).nonzero()
        scores = all_scores[batch, keep].view(-1)
        boxes = all_boxes[batch, keep, :].view(-1, 4)
        classes = all_classes[batch, keep].view(-1)

        if scores.nelement() == 0:
            continue

        # Sort boxes
        scores, indices = torch.sort(scores, descending=True)
        boxes, classes = boxes[indices], classes[indices]
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1).view(
            -1
        )
        keep = torch.ones(scores.nelement(), device=device, dtype=torch.uint8).view(-1)

        for i in range(ndetections):
            if i >= keep.nonzero().nelement() or i >= scores.nelement():
                i -= 1
                break

            # Find overlapping boxes with lower score
            xy1 = torch.max(boxes[:, :2], boxes[i, :2])
            xy2 = torch.min(boxes[:, 2:], boxes[i, 2:])
            inter = torch.prod((xy2 - xy1 + 1).clamp(0), 1)
            criterion = (
                (scores > scores[i])
                | (inter / (areas + areas[i] - inter) <= nms)
                | (classes != classes[i])
            )
            criterion[i] = 1

            # Only keep relevant boxes
            scores = scores[criterion.nonzero()].view(-1)
            boxes = boxes[criterion.nonzero(), :].view(-1, 4)
            classes = classes[criterion.nonzero()].view(-1)
            areas = areas[criterion.nonzero()].view(-1)
            keep[(~criterion).nonzero()] = 0

        out_scores[batch, : i + 1] = scores[: i + 1]
        out_boxes[batch, : i + 1, :] = boxes[: i + 1, :]
        out_classes[batch, : i + 1] = classes[: i + 1]

    return out_scores, out_boxes, out_classes


def detection_postprocess(image, cls_heads, box_heads):
    anchors = {}
    decoded = []

    for cls_head, box_head in zip(cls_heads, box_heads):
        print("stride calc: ", image.shape[-2], cls_head.shape[-2])
        stride = image.shape[-2] // cls_head.shape[-2]
        if stride not in anchors:
            anchors[stride] = generate_anchors(
                stride,
                ratio_vals=[1.0, 2.0, 0.5],
                scales_vals=[4 * 2 ** (i / 3) for i in range(3)],
            )
        decoded.append(
            decode(
                cls_head,
                box_head,
                stride,
                threshold=0.05,
                top_n=1000,
                anchors=anchors[stride],
            )
        )
    decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
    scores, boxes, labels = nms(*decoded, nms=0.5, ndetections=100)
    return scores, boxes, labels


def render_detections(
    image: Image.Image, boxes, scores, labels, score_thresh=0.5, max_det=20
):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    label_map = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "TV",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "and toothbrush",
    ]

    # Font setup (optional)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    # Convert tensors to numpy for easy iteration
    boxes = boxes[0].cpu().numpy()
    scores = scores[0].cpu().numpy()
    labels = labels[0].cpu().numpy()

    count = 0
    for i, score in enumerate(scores):
        if score < score_thresh:
            continue
        if count >= max_det:
            break

        box = boxes[i].tolist()  # [x1, y1, x2, y2]
        label = label_map[int(labels[i])]
        print(score, label, box)
        draw.rectangle(box, outline="red", width=2)
        text = f"{label}:{score:.2f}"
        text_size = draw.textlength(text, font=font)
        draw.rectangle(
            [box[0], box[1] - 18, box[0] + text_size + 4, box[1]], fill="red"
        )
        draw.text((box[0] + 2, box[1] - 18), text, fill="white", font=font)
        count += 1

    return image


def resize_with_padding(
    img: Image.Image, target_width=640, target_height=480, fill_color=(0, 0, 0)
):
    """
    Resize and pad the image to the target size (W×H) while maintaining aspect ratio.
    Pads with the specified fill_color (default black).
    """
    original_width, original_height = img.size
    target_ratio = target_width / target_height
    original_ratio = original_width / original_height

    # Compute new size preserving aspect ratio
    if original_ratio > target_ratio:  # wider than target
        new_width = target_width
        new_height = int(target_width / original_ratio)
    else:  # taller than target
        new_height = target_height
        new_width = int(target_height * original_ratio)

    # Resize
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create new image and paste resized image centered
    new_img = Image.new("RGB", (target_width, target_height), fill_color)
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2
    new_img.paste(img_resized, (pad_x, pad_y))

    return new_img


input_image = Image.open("input.jpeg")
input_image = resize_with_padding(input_image, 640, 480)

input_tensor = preprocess(input_image)
input_tensor = input_tensor.unsqueeze(0)
inputs_flatten = flatten(input_tensor.detach().cpu().numpy())
inputs_flatten = update_flatten_list(inputs_flatten, [])

sess = onnxruntime.InferenceSession("model.onnx")
ort_inputs = dict(
    (sess.get_inputs()[i].name, to_numpy(input))
    for i, input in enumerate(inputs_flatten)
)

res = sess.run(None, ort_inputs)

print(len(res))

scores, boxes, labels = detection_postprocess(
    input_tensor[0],
    [torch.from_numpy(arr) for arr in res[:5]],
    [torch.from_numpy(arr) for arr in res[5:]],
)

# Render and save the output
output_image = render_detections(input_image.copy(), boxes, scores, labels, 0.01, 3)
output_image.save("output_with_boxes.jpg")
