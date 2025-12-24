use image::{ImageBuffer, Rgb, flat::SampleLayout, imageops::FilterType};
use ndarray::{Array3, ShapeBuilder};

pub type Image = ImageBuffer<Rgb<u8>, Vec<u8>>;

pub fn resize_padded(img: Image, target_size: (u32, u32)) -> Image {
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

    let mut new_img = ImageBuffer::from_pixel(target_size.0, target_size.1, Rgb([127, 127, 127]));

    image::imageops::overlay(&mut new_img, &img_resized, pad_left, pad_top);

    new_img
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
