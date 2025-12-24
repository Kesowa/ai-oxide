use image::DynamicImage;
use ndarray::{Array, Array2, Axis, Ix4};
use ort::{session::Session, value::Tensor};

use crate::model::{
    Model,
    utils::{image_to_ndarray3, resize_padded},
};

pub struct BBox {
    pub bounds: [u32; 4],
    pub score: f32,
    pub label: i32,
}

pub struct Mobilenet {
    session: Session,
}

impl Model<f32, Ix4> for Mobilenet {
    const INPUT_SHAPE: [usize; 4] = [1, 512, 512, 3];

    type Input = DynamicImage;

    type Output = Array2<u8>;

    fn preprocess(input: Self::Input) -> Array<f32, Ix4> {
        let resized = resize_padded(
            input.into_rgb8(),
            (Self::INPUT_SHAPE[2] as u32, Self::INPUT_SHAPE[1] as u32),
        );
        let array = image_to_ndarray3(resized);
        let img = array.mapv(|v| (v as f32) / 255.0).reversed_axes();

        let stacked = img.insert_axis(Axis(0));
        stacked
    }

    fn load(session: Session) -> Result<Self, ort::Error> {
        let input_shape = session.inputs[0].input_type.tensor_shape().unwrap();

        assert_eq!(input_shape[1..], Self::INPUT_SHAPE.map(|v| v as i64)[1..]);

        Ok(Self { session })
    }

    fn postprocess(output: ort::session::SessionOutputs) -> Self::Output {
        let output: Tensor<f32> = output.into_iter().next().unwrap().1.downcast().unwrap();
        let output = output
            .extract_array()
            .to_shape((512, 512))
            .unwrap()
            .mapv(|v| (v * 255.0) as u8);
        output
    }
    fn session(&mut self) -> &mut Session {
        &mut self.session
    }
}
