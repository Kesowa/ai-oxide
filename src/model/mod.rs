use std::sync::Once;

use ndarray::Array;
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{Session, SessionOutputs, builder::GraphOptimizationLevel},
    value::Tensor,
};

pub mod clahe;
pub mod utils;

pub mod mobilenet;
pub mod retinanet;

static ORT_INIT: Once = Once::new();

pub trait Model<A, D>
where
    Self: Sized,
    A: Copy
        + 'static
        + std::fmt::Debug
        + ort::tensor::IntoTensorElementType
        + ort::tensor::PrimitiveTensorElementType,
    D: ndarray::Dimension + 'static,
{
    type Input;
    type Output;
    const INPUT_SHAPE: [usize; 4];

    fn new(model: &[u8]) -> Result<Self, ort::Error> {
        ORT_INIT.call_once(|| {
            ort::init()
                .with_execution_providers([CPUExecutionProvider::default().build()])
                .commit()
                .expect("failed to init ort");
        });
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(true)?
            .with_inter_threads(2)?
            .with_intra_threads(num_cpus::get_physical())?
            .commit_from_memory(model)?;

        Self::load(session)
    }
    fn load(session: Session) -> Result<Self, ort::Error>;
    fn preprocess(input: Self::Input) -> Array<A, D>;
    fn postprocess(outputs: SessionOutputs) -> Self::Output;
    fn run(&mut self, input: Self::Input) -> Result<Self::Output, ort::Error> {
        let input = Self::preprocess(input);
        let session = self.session();
        let input_name = session.inputs[0].name.clone();
        let outputs = session.run(ort::inputs![input_name => Tensor::from_array(input)?])?;
        let output = Self::postprocess(outputs);
        Ok(output)
    }
    fn session(&mut self) -> &mut Session;
}
