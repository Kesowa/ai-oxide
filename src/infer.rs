use std::error::Error;

use image::{imageops::FilterType, DynamicImage, ImageBuffer, Rgb};
use ndarray::Array4;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

pub struct Infer {
    session: Session,
}

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

impl Infer {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }

    pub fn infer_image(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        let resized = resize_padded(image, 512, 512);
        let image = Array4::from_shape_vec((1, 512, 512, 3), resized.into_vec())
            .expect("This should never fail");
        let mut image = image.mapv(|e| f32::from(e) / 255.0);
        image.swap_axes(2, 1);
        let tensor = Tensor::from_array(image)?;
        let outputs = self.session.run(ort::inputs![tensor]?)?;
        let generated_tags = outputs[0].try_extract_tensor::<f32>()?.flatten().to_vec();
        Ok(generated_tags)
    }
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
