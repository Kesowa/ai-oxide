use image::{DynamicImage, ImageBuffer, Rgb, flat::SampleLayout, imageops::FilterType};
use ndarray::{Array3, Axis, ShapeBuilder, s};
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};

use crate::post;

pub struct Infer {
    session: Session,
    input_size: u32,
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
            input_size = shape[2].max(shape[3]) as u32;
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
            .slice(s![.., 80..560, ..640])
            .mapv(|v| (v as f32) / 255.0)
            .reversed_axes();

        // Normalize pixels
        let mean = ndarray::arr1(&[0.485, 0.456, 0.406]);
        let std = ndarray::arr1(&[0.229, 0.224, 0.225]);
        let img = (img - mean) / std;

        let stacked = img.reversed_axes().insert_axis(Axis(0));

        let outputs = self.session.run(
            ort::inputs![input_name => TensorRef::from_array_view(&stacked.as_standard_layout())?],
        )?;
        let output = post::retinanet(outputs);
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

fn resize_padded(img: Image, target_size: u32) -> Image {
    let width = img.width();
    let height = img.height();
    let max_dim = width.max(height);
    let pad_left = (max_dim - width) / 2;
    let pad_top = (max_dim - height) / 2;

    let mut padded = ImageBuffer::from_pixel(max_dim, max_dim, Rgb([255, 255, 255]));
    image::imageops::overlay(&mut padded, &img, pad_left.into(), pad_top.into());

    if max_dim != target_size {
        padded = image::imageops::resize(&padded, target_size, target_size, FilterType::CatmullRom);
    }

    padded
}

#[test]
fn test_infer() {
    let mut model = Infer::new("./retinanet-9.onnx").unwrap();
    let inference = model
        .infer_image(image::open("./test.JPEG").unwrap())
        .unwrap();
    println!("Length: {}", inference.len());
    inference
        .iter()
        .filter(|b| b.bounds.iter().any(|&v| v > 0))
        .for_each(|b| {
            println!("{b:?}");
        });
}
