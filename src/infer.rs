#[cfg(test)]
mod test {
    #[test]
    fn test_retinanet() {
        static LABEL_MAP: [&str; 80] = [
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
        ];

        use crate::model::{Model, retinanet::Retinanet, utils::*};
        use imageproc::{drawing::draw_hollow_rect_mut, rect::Rect};
        let model_data = std::fs::read("./retinanet-9.onnx").unwrap();
        let mut model = Retinanet::new(&model_data).unwrap();
        let img = image::open("./test1.JPEG").unwrap();
        let inference = model.run(img.clone()).unwrap();
        let mut img = resize_padded(
            img.into(),
            (
                Retinanet::INPUT_SHAPE[3] as u32,
                Retinanet::INPUT_SHAPE[2] as u32,
            ),
        );
        for (i, item) in inference.iter().enumerate() {
            if item.score < 0.1 || i > 3 {
                break;
            };
            println!("{}: {:.2}", LABEL_MAP[item.label as usize], item.score);
            println!("BBOX: {:?}", item.bounds);
            let bbox = item.bounds;
            draw_hollow_rect_mut(
                &mut img,
                Rect::at(bbox[0] as i32, bbox[1] as i32)
                    .of_size(bbox[2] - bbox[0], bbox[3] - bbox[1]),
                image::Rgb([255u8, 0u8, 0u8]),
            );
        }
        img.save("output.png").unwrap();
    }

    #[test]
    fn test_mobilenet() {
        use crate::model::{Model, mobilenet::Mobilenet};
        use image::{ImageBuffer, Luma};

        let model_data = std::fs::read("./model_RS.onnx").unwrap();
        let mut model = Mobilenet::new(&model_data).unwrap();
        let img = image::open("./rooftop.png").unwrap();
        let inference = model.run(img.clone()).unwrap();
        let mask: ImageBuffer<Luma<u8>, _> = ImageBuffer::from_raw(
            Mobilenet::INPUT_SHAPE[2] as u32,
            Mobilenet::INPUT_SHAPE[1] as u32,
            inference
                .flatten_with_order(ndarray::Order::RowMajor)
                .to_vec(),
        )
        .unwrap();
        // let mask: ImageBuffer<Rgb<u8>, Vec<_>> = mask.convert();
        // let mut img = resize_padded(
        //     img.into(),
        //     (
        //         Mobilenet::INPUT_SHAPE[2] as u32,
        //         Mobilenet::INPUT_SHAPE[1] as u32,
        //     ),
        // );
        // imageops::overlay(&mut img, &mask, 0, 0);
        mask.save("roof_out.png").unwrap();
    }
}
