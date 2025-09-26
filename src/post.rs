use std::collections::BTreeMap;

use ndarray::{Array1, Array2, Array3, Array4, Axis, Ix4};
use ort::{session::SessionOutputs, value::Tensor};

pub struct Box {
    bounds: [f32; 4],
    score: f32,
    label: f32,
}

pub fn retinanet(output: SessionOutputs) -> Vec<Box> {
    let img_height = 640;
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
        let stride = img_height / cls_head.shape().last().unwrap();
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
        println!(
            "stride: {stride}, class: {:?}; regress: {:?}",
            cls_head.shape(),
            reg_head.shape(),
        );
    }
    println!("anchors: {anchors:?}");
    todo!();
}

/// Generate anchors coordinates [x1, y1, x2, y2] from stride, ratios, and scales
pub fn generate_anchors(stride: f32, ratio_vals: &[f32], scales_vals: &[f32]) -> Array2<f32> {
    let num_ratios = ratio_vals.len();
    let num_scales = scales_vals.len();
    let num_anchors = num_ratios * num_scales;

    // Expand ratios and scales to match like PyTorch repeat
    let mut ratios: Vec<f32> = Vec::with_capacity(num_anchors);
    for _ in 0..num_scales {
        for &r in ratio_vals {
            ratios.push(r);
        }
    }

    let mut scales: Vec<f32> = Vec::with_capacity(num_anchors);
    for &s in scales_vals {
        for _ in 0..num_ratios {
            scales.push(s);
        }
    }

    let ratios = Array1::from(ratios);
    let scales = Array1::from(scales);

    // Base box size = stride x stride
    let wh = Array1::from(vec![stride; num_anchors]);

    // Compute widths and heights per anchor
    let ws = (&wh * &wh / &ratios).mapv(f32::sqrt);
    let hs = &ws * &ratios;

    // Apply scales
    let ws_scaled = &ws * &scales;
    let hs_scaled = &hs * &scales;

    // Compute x1,y1,x2,y2 (centered at stride/2, stride/2)
    let x1 = (&wh - &ws_scaled) * 0.5;
    let y1 = (&wh - &hs_scaled) * 0.5;
    let x2 = (&wh + &ws_scaled) * 0.5;
    let y2 = (&wh + &hs_scaled) * 0.5;

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
        let ratios = vec![0.5, 1.0, 2.0];
        let scales = vec![1.0, 2.0];

        let anchors = generate_anchors(stride, &ratios, &scales);
        println!("{:?}", anchors);

        assert_eq!(anchors.shape(), &[ratios.len() * scales.len(), 4]);
    }
}

// src/decode.rs

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
            let class_id = ((*idx / (h * w)) % num_classes) as i32;
            let x = (*idx % w) as i32;
            let y = ((*idx / w) % h) as i32;
            let a = (*idx / (num_classes * h * w)) as usize;

            // Fetch box deltas [4]
            let mut deltas = Array2::<f32>::zeros((1, 4));
            for k in 0..4 {
                deltas[[0, k]] = box_flat[[a * 4 + k, y as usize, x as usize]];
            }

            // Anchor
            let mut anchor = Array2::<f32>::zeros((1, 4));
            for k in 0..4 {
                anchor[[0, k]] = anchors[[a, k]]
                    + if k < 2 {
                        (if k % 2 == 0 { x } else { y }) as f32 * stride as f32
                    } else {
                        (if k % 2 == 0 { x } else { y }) as f32 * stride as f32
                    };
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
