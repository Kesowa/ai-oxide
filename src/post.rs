use std::collections::BTreeMap;

use ndarray::{Array, Array1, Array2, Array3, Array4, Axis, Ix4, concatenate, s};
use ort::{session::SessionOutputs, value::Tensor};

#[derive(Debug)]
pub struct Box {
    pub bounds: [u32; 4],
    pub score: f32,
    pub label: i32,
}

pub fn retinanet(img_shape: &[usize], output: SessionOutputs) -> Vec<Box> {
    let ratio_vals = [1.0, 2.0, 0.5];
    let scales_vals = (0..3)
        .map(|i| 4. * 2.0f32.powf(i as f32 / 3.))
        .collect::<Vec<_>>();
    let mut cls_heads = output
        .into_iter()
        .map(|(_, v)| {
            let tensor: Tensor<f32> = v.downcast().unwrap();
            let array = tensor.extract_array();
            let array4 = array.into_dimensionality::<Ix4>().unwrap();
            array4.to_owned()
        })
        .collect::<Vec<_>>();
    let reg_heads = cls_heads.split_off(5);
    let mut anchors = BTreeMap::new();
    let mut decoded = Vec::with_capacity(cls_heads.len());
    for (cls_head, reg_head) in cls_heads.iter().zip(reg_heads.iter()) {
        let stride = img_shape.iter().rev().skip(1).next().unwrap()
            / cls_head.shape().iter().rev().skip(1).next().unwrap();
        if anchors.get(&stride).is_none() {
            anchors.insert(
                stride,
                generate_anchors(stride as f32, &ratio_vals, &scales_vals),
            );
        };
        decoded.push(decode(
            &cls_head,
            &reg_head,
            &anchors[&stride],
            stride,
            0.05,
            1000,
        ));
    }
    let all_scores = concatenate(
        Axis(1),
        &decoded.iter().map(|d| d.0.view()).collect::<Vec<_>>(),
    )
    .unwrap();
    let all_boxes = concatenate(
        Axis(1),
        &decoded.iter().map(|d| d.1.view()).collect::<Vec<_>>(),
    )
    .unwrap();
    let all_classes = concatenate(
        Axis(1),
        &decoded.iter().map(|d| d.2.view()).collect::<Vec<_>>(),
    )
    .unwrap();
    let (scores, boxes, labels) = nms(&all_scores, &all_boxes, &all_classes, 0.5, 100);
    let mut bboxes = Vec::with_capacity(100);
    for i in 0..100 {
        bboxes.push(Box {
            bounds: [
                (boxes[(0, i, 0)]).ceil() as u32,
                (boxes[(0, i, 1)]).ceil() as u32,
                (boxes[(0, i, 2)]).ceil() as u32,
                (boxes[(0, i, 3)]).ceil() as u32,
            ],
            score: scores[(0, i)],
            label: labels[(0, i)],
        });
    }
    bboxes
}

/// Generate anchors coordinates [x1, y1, x2, y2] from stride, ratios, and scales
pub fn generate_anchors(stride: f32, ratio_vals: &[f32], scales_vals: &[f32]) -> Array2<f32> {
    let num_ratios = ratio_vals.len();
    let num_scales = scales_vals.len();
    let num_anchors = num_ratios * num_scales;

    let mut ratios = Vec::with_capacity(num_anchors);
    let mut scales = Vec::with_capacity(num_anchors);
    
    for scale in scales_vals {
        for ratio in ratio_vals {
            scales.push(*scale);
            ratios.push(*ratio);
        }
    }
    
    let ratios = Array1::from(ratios);
    let scales = Array1::from(scales);

    // Base box size = stride x stride
    let wh = stride;

    // Compute widths and heights per anchor
    let ws = ratios.mapv(|r| (wh * wh / r).sqrt());
    let hs = &ws * &ratios;

    // Apply scales
    let ws_scaled = &ws * &scales;
    let hs_scaled = &hs * &scales;

    // Compute x1,y1,x2,y2 (centered at stride/2, stride/2)
    let x1 = (wh - &ws_scaled) * 0.5;
    let y1 = (wh - &hs_scaled) * 0.5;
    let x2 = (wh + &ws_scaled) * 0.5;
    let y2 = (wh + &hs_scaled) * 0.5;

    // Stack into [num_anchors, 4]
    let mut anchors = Array2::<f32>::zeros((num_anchors, 4));
    for i in 0..num_anchors {
        anchors[[i, 0]] = x1[i];
        anchors[[i, 1]] = y1[i];
        anchors[[i, 2]] = x2[i];
        anchors[[i, 3]] = y2[i];
    }

    anchors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_anchors() {
        let stride = 32.0;
        let ratios = vec![1.0, 2.0, 0.5];
        let scales = vec![4.0 * 2.0f32.powf(0.0/3.0), 4.0 * 2.0f32.powf(1.0/3.0), 4.0 * 2.0f32.powf(2.0/3.0)];

        let anchors = generate_anchors(stride, &ratios, &scales);
        println!("{:?}", anchors);

        assert_eq!(anchors.shape(), &[ratios.len() * scales.len(), 4]);
    }
}

/// Convert deltas from anchors to boxes
pub fn delta2box(
    deltas: &Array2<f32>,  // [N, 4]
    anchors: &Array2<f32>, // [N, 4]
    size: (usize, usize),  // (W, H)
    stride: f32,
) -> Array2<f32> {
    let n = deltas.shape()[0];
    let mut boxes = Array2::<f32>::zeros((n, 4));

    for i in 0..n {
        let dx = deltas[[i, 0]];
        let dy = deltas[[i, 1]];
        let dw = deltas[[i, 2]];
        let dh = deltas[[i, 3]];

        let ax1 = anchors[[i, 0]];
        let ay1 = anchors[[i, 1]];
        let ax2 = anchors[[i, 2]];
        let ay2 = anchors[[i, 3]];

        let aw = ax2 - ax1 + 1.0;
        let ah = ay2 - ay1 + 1.0;
        let ctr_x = ax1 + 0.5 * aw;
        let ctr_y = ay1 + 0.5 * ah;

        let pred_ctr_x = dx * aw + ctr_x;
        let pred_ctr_y = dy * ah + ctr_y;
        let pred_w = dw.exp() * aw;
        let pred_h = dh.exp() * ah;

        let mut x1 = pred_ctr_x - 0.5 * pred_w;
        let mut y1 = pred_ctr_y - 0.5 * pred_h;
        let mut x2 = pred_ctr_x + 0.5 * pred_w - 1.0;
        let mut y2 = pred_ctr_y + 0.5 * pred_h - 1.0;

        // Clamp
        let max_w = (size.0 as f32) * stride - 1.0;
        let max_h = (size.1 as f32) * stride - 1.0;
        x1 = x1.clamp(0.0, max_w);
        y1 = y1.clamp(0.0, max_h);
        x2 = x2.clamp(0.0, max_w);
        y2 = y2.clamp(0.0, max_h);

        boxes[[i, 0]] = x1;
        boxes[[i, 1]] = y1;
        boxes[[i, 2]] = x2;
        boxes[[i, 3]] = y2;
    }

    boxes
}

/// Box Decoding and Filtering
pub fn decode(
    all_cls_head: &Array4<f32>, // [B, A*C, H, W]
    all_box_head: &Array4<f32>, // [B, A*4, H, W]
    anchors: &Array2<f32>,
    stride: usize,
    threshold: f32,
    top_n: usize,
) -> (Array2<f32>, Array3<f32>, Array2<i32>) {
    let (batch_size, ac, h, w) = (
        all_cls_head.shape()[0],
        all_cls_head.shape()[1],
        all_cls_head.shape()[2],
        all_cls_head.shape()[3],
    );
    let num_anchors = anchors.shape()[0];
    let num_classes = ac / num_anchors;

    // Output placeholders
    let mut out_scores = Array2::<f32>::zeros((batch_size, top_n));
    let mut out_boxes = Array3::<f32>::zeros((batch_size, top_n, 4));
    let mut out_classes = Array2::<i32>::zeros((batch_size, top_n));

    for b in 0..batch_size {
        // Flatten class scores
        let cls_flat = all_cls_head
            .index_axis(Axis(0), b)
            .to_owned()
            .into_shape_with_order((ac * h * w,))
            .unwrap();
        let box_flat = all_box_head
            .index_axis(Axis(0), b)
            .to_owned()
            .into_shape_with_order((num_anchors * 4, h, w))
            .unwrap();

        // Keep scores above threshold
        let mut keep_indices: Vec<usize> = cls_flat
            .indexed_iter()
            .filter(|(_, val)| **val >= threshold)
            .map(|(i, _)| i)
            .collect();

        if keep_indices.is_empty() {
            continue;
        }

        // Select top N scores
        keep_indices.sort_by(|&i, &j| cls_flat[j].partial_cmp(&cls_flat[i]).unwrap());
        let topk = keep_indices.into_iter().take(top_n).collect::<Vec<_>>();

        // Gather
        for (rank, idx) in topk.iter().enumerate() {
            let score = cls_flat[*idx];

            let c = *idx / (h * w);
            let a = c / num_classes; 
            let class_id = (c % num_classes) as i32;
            
            let x = (*idx % w) as i32;
            let y = ((*idx / w) % h) as i32;

            // Fetch box deltas [4]
            let mut deltas = Array2::<f32>::zeros((1, 4));
            for k in 0..4 {
                deltas[[0, k]] = box_flat[[a * 4 + k, y as usize, x as usize]];
            }

            // Anchor - add grid offset
            let mut anchor = Array2::<f32>::zeros((1, 4));
            for k in 0..4 {
                anchor[[0, k]] =
                    anchors[[a, k]] + (if k % 2 == 0 { x } else { y }) as f32 * stride as f32;
            }

            let decoded = delta2box(&deltas, &anchor, (w, h), stride as f32);

            out_scores[[b, rank]] = score;
            for k in 0..4 {
                out_boxes[[b, rank, k]] = decoded[[0, k]];
            }
            out_classes[[b, rank]] = class_id;
        }
    }

    (out_scores, out_boxes, out_classes)
}

pub fn nms(
    all_scores: &Array2<f32>,  // [B, N]
    all_boxes: &Array3<f32>,   // [B, N, 4]
    all_classes: &Array2<i32>, // [B, N]
    nms_thresh: f32,
    ndetections: usize,
) -> (Array2<f32>, Array3<f32>, Array2<i32>) {
    let batch_size = all_scores.shape()[0];
    let num_boxes = all_scores.shape()[1];

    // Outputs
    let mut out_scores = Array2::<f32>::zeros((batch_size, ndetections));
    let mut out_boxes = Array3::<f32>::zeros((batch_size, ndetections, 4));
    let mut out_classes = Array2::<i32>::zeros((batch_size, ndetections));

    for b in 0..batch_size {
        // Collect valid boxes
        let mut scores: Vec<(usize, f32)> = (0..num_boxes)
            .filter_map(|i| {
                let s = all_scores[[b, i]];
                if s > 0.0 { Some((i, s)) } else { None }
            })
            .collect();

        if scores.is_empty() {
            continue;
        }

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Build vectors
        let mut boxes: Vec<[f32; 4]> = scores
            .iter()
            .map(|&(i, _)| {
                let row = all_boxes.slice(s![b, i, ..]);
                [row[0], row[1], row[2], row[3]]
            })
            .collect();

        let mut cls: Vec<i32> = scores.iter().map(|&(i, _)| all_classes[[b, i]]).collect();
        let mut scr: Vec<f32> = scores.iter().map(|&(_, s)| s).collect();

        let mut kept_scores = Vec::new();
        let mut kept_boxes = Vec::new();
        let mut kept_classes = Vec::new();

        let mut i = 0;
        while i < scr.len() && kept_scores.len() < ndetections {
            // Reference box
            let ref_box = boxes[i];
            let ref_cls = cls[i];
            let ref_score = scr[i];

            kept_scores.push(ref_score);
            kept_boxes.push(ref_box);
            kept_classes.push(ref_cls);

            // Filter rest
            let mut new_boxes = Vec::new();
            let mut new_scores = Vec::new();
            let mut new_classes = Vec::new();

            for j in (i + 1)..scr.len() {
                let iou = iou(ref_box, boxes[j]);
                if cls[j] != ref_cls || iou <= nms_thresh {
                    new_boxes.push(boxes[j]);
                    new_scores.push(scr[j]);
                    new_classes.push(cls[j]);
                }
            }

            boxes = new_boxes;
            scr = new_scores;
            cls = new_classes;
            i = 0; // restart from next top
        }

        let n_keep = kept_scores.len().min(ndetections);

        for k in 0..n_keep {
            out_scores[[b, k]] = kept_scores[k];
            out_boxes
                .slice_mut(s![b, k, ..])
                .assign(&Array::from_vec(kept_boxes[k].to_vec()));
            out_classes[[b, k]] = kept_classes[k];
        }
    }

    (out_scores, out_boxes, out_classes)
}

/// Compute IoU between two boxes [x1, y1, x2, y2]
fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let (x1, y1, x2, y2) = (
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    );
    let inter_w = (x2 - x1 + 1.0).max(0.0);
    let inter_h = (y2 - y1 + 1.0).max(0.0);
    let inter = inter_w * inter_h;

    let area_a = (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0);
    let area_b = (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0);

    inter / (area_a + area_b - inter)
}