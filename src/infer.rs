use image::{DynamicImage, ImageBuffer, Rgb, flat::SampleLayout, imageops::FilterType};
use ndarray::{Array3, Axis, ShapeBuilder};
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};

use crate::post;

pub struct Infer {
    session: Session,
    input_size: (u32, u32),
}

pub type Image = ImageBuffer<Rgb<u8>, Vec<u8>>;

impl Infer {
    pub fn new(model_path: &str) -> Result<Self, ort::Error> {
        ort::init()
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .commit()?;
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(true)?
            .with_inter_threads(2)?
            .with_intra_threads(num_cpus::get_physical())?
            .commit_from_file(model_path)?;
        let input = &session.inputs[0];
        let input_size;
        if let Some(shape) = input.input_type.tensor_shape()
            && shape[1] == 3
        // && shape[2] == shape[3]
        {
            input_size = (shape[3] as u32, shape[2] as u32);
        } else {
            return Err(ort::Error::new_with_code(
                ort::ErrorCode::GenericFailure,
                format!("invalid input shape: {:?}", input.input_type.tensor_shape()),
            ));
        }
        Ok(Self {
            session,
            input_size,
        })
    }

    pub fn infer_image(&mut self, image: DynamicImage) -> Result<Vec<post::Box>, ort::Error> {
        let resized = resize_padded(image.into_rgb8(), self.input_size);
        let array = image_to_ndarray3(resized);
        self.infer(array)
    }

    pub fn infer(&mut self, array: Array3<u8>) -> Result<Vec<post::Box>, ort::Error> {
        let input_name = self.session.inputs[0].name.clone();
        // let output_name = self.session.outputs[0].name.clone();
        let img = array
            // .slice(s![.., ..640, ..480])
            .mapv(|v| (v as f32) / 255.0)
            .reversed_axes();

        let img_shape = img.shape().to_vec();

        // Normalize pixels
        let mean = ndarray::arr1(&[0.485, 0.456, 0.406]);
        let std = ndarray::arr1(&[0.229, 0.224, 0.225]);
        let img = (img - mean) / std;

        let stacked = img.reversed_axes().insert_axis(Axis(0));

        let outputs = self.session.run(
            ort::inputs![input_name => TensorRef::from_array_view(&stacked.as_standard_layout())?],
        )?;
        let output = post::retinanet(&img_shape, outputs);
        Ok(output)
    }
}

pub fn image_to_ndarray3(image: Image) -> Array3<u8> {
    let SampleLayout {
        channels,
        channel_stride,
        height,
        height_stride,
        width,
        width_stride,
    } = image.sample_layout();
    let shape = (channels as usize, height as usize, width as usize);
    let strides = (channel_stride, height_stride, width_stride);
    Array3::from_shape_vec(shape.strides(strides), image.into_raw()).unwrap()
}

pub fn ndarray3_to_image(array: Array3<u8>) -> Image {
    let shape = array.shape();
    let width = shape[2];
    let height = shape[1];
    let arr = array.permuted_axes((1, 2, 0));
    image::ImageBuffer::from_raw(
        width as u32,
        height as u32,
        arr.flatten_with_order(ndarray::Order::RowMajor).to_vec(),
    )
    .unwrap()
}

fn resize_padded(img: Image, target_size: (u32, u32)) -> Image {
    let original_width = img.width() as f32;
    let original_height = img.height() as f32;
    let target_width = target_size.0 as f32;
    let target_height = target_size.1 as f32;

    let target_ratio = target_width / target_height;
    let original_ratio = original_width / original_height;

    let (new_width, new_height) = if original_ratio > target_ratio {
        (target_width, target_width / original_ratio)
    } else {
        (target_height * original_ratio, target_height)
    };

    let img_resized = image::imageops::resize(
        &img,
        new_width.round() as u32,
        new_height.round() as u32,
        FilterType::Lanczos3,
    );

    let pad_left = ((target_width - new_width) / 2.0).round() as i64;
    let pad_top = ((target_height - new_height) / 2.0).round() as i64;

    let mut new_img = ImageBuffer::from_pixel(target_size.0, target_size.1, Rgb([0, 0, 0]));

    image::imageops::overlay(&mut new_img, &img_resized, pad_left, pad_top);

    new_img
}

#[test]
fn test_infer() {
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

    use imageproc::{drawing::draw_hollow_rect_mut, rect::Rect};
    let mut model = Infer::new("./retinanet-9.onnx").unwrap();
    let img = image::open("./input.jpeg").unwrap();
    let inference = model.infer_image(img.clone()).unwrap();
    let mut img = resize_padded(img.into(), model.input_size);
    for (i, item) in inference.iter().enumerate() {
        if item.score < 0.1 || i > 3 {
            break;
        };
        println!("{}: {:.2}", LABEL_MAP[item.label as usize], item.score);
        println!("BBOX: {:?}", item.bounds);
        let bbox = item.bounds;
        draw_hollow_rect_mut(
            &mut img,
            Rect::at(bbox[0] as i32, bbox[1] as i32).of_size(bbox[2] - bbox[0], bbox[3] - bbox[1]),
            image::Rgb([255u8, 0u8, 0u8]),
        );
    }
    img.save("output.png").unwrap();
}