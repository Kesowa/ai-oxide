use std::error::Error;

use image::{
    flat::SampleLayout, imageops::FilterType, DynamicImage, ImageBuffer, Pixel, Rgb, Rgba,
};
use ndarray::{Array3, Array4, ArrayView3, ShapeBuilder};
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

pub struct Infer {
    session: Session,
}

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

impl Infer {
    pub fn new(model_path: &str) -> Result<Self> {
        ort::init()
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .commit()?;
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;
        println!("{:#?}", session.inputs[0]);
        Ok(Self { session })
    }

    pub fn infer_image(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        let resized = resize_padded(image, 224, 224);
        let image = Array4::from_shape_vec((1, 224, 224, 3), resized.into_vec())
            .expect("This should never fail");
        let image = image.permuted_axes([0, 3, 1, 2]);
        let tensor = Tensor::from_array(image)?;
        let outputs = self.session.run(ort::inputs![tensor]?)?;
        let generated_tags = outputs[0].try_extract_tensor::<f32>()?.flatten().to_vec();
        Ok(generated_tags)
    }

    pub fn infer(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        let resized = resize_padded(image, 224, 224);
        let image = Array4::from_shape_vec((1, 224, 224, 3), resized.into_vec())
            .expect("This should never fail");
        let image = image.permuted_axes([0, 3, 1, 2]);
        let tensor = Tensor::from_array(image)?;
        let outputs = self.session.run(ort::inputs![tensor]?)?;
        let generated_tags = outputs[0].try_extract_tensor::<f32>()?.flatten().to_vec();
        Ok(generated_tags)
    }
}

pub fn image_to_ndarray3<P: Pixel + 'static>(
    image: ImageBuffer<P, Vec<P::Subpixel>>,
) -> Array3<P::Subpixel> {
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

pub fn ndarray3_to_image(array: Array3<u8>) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let shape = array.shape();
    let width = shape[1];
    let height = shape[2];
    let arr = array.permuted_axes((1, 2, 0));
    image::ImageBuffer::from_raw(
        width as u32,
        height as u32,
        arr.flatten_with_order(ndarray::Order::RowMajor)
            .as_slice()
            .unwrap()
            .to_vec(),
    )
    .unwrap()
}

fn resize_padded(
    img: &DynamicImage,
    max_width: u32,
    max_height: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut width = img.width();
    let mut height = img.height();
    let aspect_ratio = (width as f32) / (height as f32);

    if width > max_width || height < max_height {
        width = max_width;
        height = ((width as f32) / aspect_ratio) as u32;
    }

    if height > max_height || width < max_width {
        height = max_height;
        width = ((height as f32) * aspect_ratio) as u32;
    }

    let thumbnail = img.resize_exact(width, height, FilterType::Gaussian);
    let mut img = ImageBuffer::from_pixel(max_width, max_height, Rgb([255, 255, 255]));
    image::imageops::overlay(
        &mut img,
        &thumbnail.to_rgb8(),
        (max_width - width) as i64 / 2,
        (max_height - height) as i64 / 2,
    );
    img
}

#[test]
fn test_infer() {
    let model = Infer::new("./mobilenetv2-7.onnx").unwrap();
    let inference = model
        .infer_image(&image::open("./output/1536.2048.png").unwrap())
        .unwrap();
    println!("{}", inference.len());
}
