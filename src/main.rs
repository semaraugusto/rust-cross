#![no_main]
sp1_zkvm::entrypoint!(main);

mod utils;

use anyhow::Error as E;
use candle::{DType, Device, Result, Shape, Tensor, D};
use candle_nn::{loss, ops, Conv2d, Linear, Module, ModuleT, Optimizer, VarBuilder, VarMap};
use std::collections::HashMap;

const IMAGE_DIM: usize = 784;
const LABELS: usize = 10;

fn linear_z(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Linear> {
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", candle_nn::init::ZERO)?;
    let bs = vs.get_with_hints(out_dim, "bias", candle_nn::init::ZERO)?;
    Ok(Linear::new(ws, Some(bs)))
}

trait Model: Sized {
    fn new(vs: VarBuilder) -> Result<Self>;
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

struct LinearModel {
    linear: Linear,
}

impl Model for LinearModel {
    fn new(vs: VarBuilder) -> Result<Self> {
        let linear = linear_z(IMAGE_DIM, LABELS, vs)?;
        Ok(Self { linear })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear.forward(xs)
    }
}

struct Mlp {
    ln1: Linear,
    ln2: Linear,
}

impl Model for Mlp {
    fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(IMAGE_DIM, 100, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(100, LABELS, vs.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        self.ln2.forward(&xs)
    }
}

#[derive(Debug)]
struct ConvNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    dropout: candle_nn::Dropout,
}

impl ConvNet {
    fn new(vs: VarBuilder) -> Result<Self> {
        let conv1 = candle_nn::conv2d(1, 32, 5, Default::default(), vs.pp("c1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 5, Default::default(), vs.pp("c2"))?;
        let fc1 = candle_nn::linear(1024, 1024, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(1024, LABELS, vs.pp("fc2"))?;
        let dropout = candle_nn::Dropout::new(0.5);
        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
            dropout,
        })
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (b_sz, _img_dim) = xs.dims2()?;
        let xs = xs
            .reshape((b_sz, 1, 28, 28))?
            .apply(&self.conv1)?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .max_pool2d(2)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            .relu()?;
        self.dropout.forward_t(&xs, train)?.apply(&self.fc2)
    }
}

#[derive(Debug)]
struct TrainingArgs {
    learning_rate: f64,
    load: Option<String>,
    save: Option<String>,
    epochs: usize,
}

// fn training_loop<M: Model>(
//     m: candle_datasets::vision::Dataset,
//     args: &TrainingArgs,
// ) -> anyhow::Result<()> {
//     let dev = candle::Device::cuda_if_available(0)?;
//
//     let train_labels = m.train_labels;
//     let train_images = m.train_images.to_device(&dev)?;
//     let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
//
//     let mut varmap = VarMap::new();
//     let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
//     let model = M::new(vs.clone())?;
//
//     if let Some(load) = &args.load {
//         println!("loading weights from {load}");
//         varmap.load(load)?
//     }
//
//     let mut sgd = candle_nn::SGD::new(varmap.all_vars(), args.learning_rate)?;
//     let test_images = m.test_images.to_device(&dev)?;
//     let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
//     for epoch in 1..args.epochs {
//         let logits = model.forward(&train_images)?;
//         let log_sm = ops::log_softmax(&logits, D::Minus1)?;
//         let loss = loss::nll(&log_sm, &train_labels)?;
//         sgd.backward_step(&loss)?;
//
//         let test_logits = model.forward(&test_images)?;
//         let sum_ok = test_logits
//             .argmax(D::Minus1)?
//             .eq(&test_labels)?
//             .to_dtype(DType::F32)?
//             .sum_all()?
//             .to_scalar::<f32>()?;
//         let test_accuracy = sum_ok / test_labels.dims1()? as f32;
//         println!(
//             "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
//             loss.to_scalar::<f32>()?,
//             100. * test_accuracy
//         );
//     }
//     if let Some(save) = &args.save {
//         println!("saving trained weights in {save}");
//         varmap.save(save)?
//     }
//     Ok(())
// }

#[derive(Clone)]
enum WhichModel {
    Linear,
    Mlp,
}

#[cfg(not(target_os = "zkvm"))]
// pub fn load_model(config: Config, device: &Device) -> StableLM {
pub fn load_model(device: &Device) -> LinearModel {
    let filenames = ["linear.safetensors"];
    let dtype = DType::F32;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device).unwrap() };
    Model::new(vb).unwrap()
}

#[cfg(target_os = "zkvm")]
// #[cfg(not(target_os="zkvm"))]
// pub fn load_model() -> Vec<u8> {
pub fn load_model(device: &Device) -> LinearModel {
    // pub fn load_model_2(config: Config, device: &Device) -> Result<StableLM> {
    let dtype = DType::F32;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    let starting_model_addr = 270557184usize;
    let mut addr = starting_model_addr;
    let magic = utils::read_numeric::<u32>(addr);
    addr += std::mem::size_of::<u32>();
    println!("[HERE] VALUE:: `{:?}`", magic);
    assert_eq!(magic, 0x67676D6C);
    // let model_len = read_numeric::<u64>(addr);
    // println!("[HERE] VALUE+4:: `{:?}`", model_len);
    // let model_addr = model_addr + 4;
    // let model_ptr = model_addr as *mut u8;
    let num_tensors = utils::read_numeric::<u32>(addr);
    addr += std::mem::size_of::<u32>();
    println!("[HERE] num_tensors:: `{:?}`", num_tensors);
    for _ in 0..num_tensors {
        let string_len = utils::read_numeric::<u32>(addr);
        addr += std::mem::size_of::<u32>();
        println!("[HERE] string_len: `{:?}`", string_len);
        let raw_bytes =
            unsafe { std::slice::from_raw_parts(addr as *const u8, string_len as usize) };
        addr += string_len as usize;
        let tensor_name = String::from_utf8_lossy(raw_bytes).to_string();
        println!("tensor_name: `{:?}`", tensor_name);
        println!("name_len: `{:?}`", tensor_name.len());
        let tensor_dims = utils::read_numeric::<u32>(addr);
        addr += std::mem::size_of::<u32>();
        // let mut tensor_shape = vec![];
        let mut tensor_shape = vec![];
        let mut tensor_num_elems = 1;
        for _ in 0..tensor_dims {
            let shape_i = utils::read_numeric::<usize>(addr);
            tensor_num_elems *= shape_i;
            addr += std::mem::size_of::<usize>();
            tensor_shape.push(shape_i)
        }
        let tensor_byte_len = tensor_num_elems * std::mem::size_of::<f32>() as usize;
        println!("tensor_shape!: `{:?}`", tensor_shape);
        let tensor_ptr = addr as *mut u8;
        let tensor_bytes =
            // unsafe { std::slice::from_raw_parts(addr as *const u8, tensor_byte_len as usize) };
            unsafe { Vec::from_raw_parts(tensor_ptr, tensor_byte_len as usize, tensor_byte_len as usize) };
        addr += tensor_byte_len as usize;
        println!("tensor starting bytes: `{:?}`", &tensor_bytes[..12]);
        println!(
            "tensor end bytes: `{:?}`",
            &tensor_bytes[tensor_bytes.len() - 12..]
        );
        let tensor =
            Tensor::from_raw_buffer(&tensor_bytes, DType::F32, &tensor_shape, device).unwrap();
        // println!("tensor sum: `{}`", tensor.sum(0).unwrap());
        println!("tensor: `{:?}`", tensor);
        println!("tensor debug: `{:?}`", tensor);
        tensors.insert(tensor_name, tensor);
    }

    println!("tensor keys: `{:?}`", tensors.keys());

    let vb = VarBuilder::from_tensors(tensors, dtype, device);
    LinearModel::new(vb).unwrap()
}

#[cfg(not(target_os = "zkvm"))]
pub fn load_input(device: &Device) -> Tensor {
    let m = candle_datasets::vision::mnist::load().unwrap();
    println!("train-images: {:?}", m.train_images.shape());
    println!("train-labels: {:?}", m.train_labels.shape());
    println!("test-images: {:?}", m.test_images.shape());
    println!("test-labels: {:?}", m.test_labels.shape());
    let image = m.test_images.get(0).unwrap().unsqueeze(0).unwrap();
    println!("image: {:?}", image.shape());
    image
}

// #[cfg(not(target_os="zkvm"))]
// pub fn load_model() -> Vec<u8> {
#[cfg(target_os = "zkvm")]
pub fn load_input(device: &Device) -> Tensor {
    // pub fn load_model_2(config: Config, device: &Device) -> Result<StableLM> {
    // let dtype = DType::F32;

    let starting_input_addr = 0x10000000usize;
    let mut addr = starting_input_addr;
    let shape_0 = utils::read_numeric::<usize>(addr);
    addr += std::mem::size_of::<usize>();
    let shape_1 = utils::read_numeric::<usize>(addr);
    let tensor_byte_len = shape_0 * shape_1;
    addr += std::mem::size_of::<usize>();

    let tensor_ptr = addr as *mut u8;
    let tensor_bytes =
        // unsafe { std::slice::from_raw_parts(addr as *const u8, tensor_byte_len as usize) };
        unsafe { Vec::from_raw_parts(tensor_ptr, tensor_byte_len as usize, tensor_byte_len as usize) };
    addr += tensor_byte_len as usize; // NOTE: NOT NEEDED.
    println!("tensor starting bytes: `{:?}`", &tensor_bytes[..12]);
    println!(
        "tensor end bytes: `{:?}`",
        &tensor_bytes[tensor_bytes.len() - 12..]
    );
    println!("tensor byte_len bytes: `{:?}`", tensor_byte_len);

    // let mut normalized = Vec::with_capacity(tensor_byte_len); // NOTE: VEC::WITH_CAPACITY IS BUGGED.
    let mut normalized = vec![];
    for (_, byte) in tensor_bytes.iter().enumerate() {
        normalized.push(*byte as f32 / 255.0);
        // normalized[i] = (*byte as f32 / 255.0);
    }

    // Tensor::from_raw_buffer(&tensor_bytes, DType::U8, &[shape_0, shape_1], device).unwrap()
    // Tensor::from_raw_buffer(&tensor_bytes, DType::U8, &[1, tensor_byte_len], device).unwrap()
    Tensor::from_vec(normalized, &[1, tensor_byte_len], device).unwrap()
}

struct Args {
    model: WhichModel,
}

// pub fn main() -> anyhow::Result<()> {
pub fn main() {
    // let args = Args::parse();
    println!("Starting");
    let args = Args {
        model: WhichModel::Linear,
    };
    let device = Device::Cpu;
    let model = load_model(&device);
    let input = load_input(&device);

    // let training_args = TrainingArgs {
    //     epochs: args.epochs,
    //     learning_rate: args.learning_rate.unwrap_or(default_learning_rate),
    //     load: args.load,
    //     save: args.save,
    // };
    // println!("training_args: `{:?}`", training_args);
    let output = model.forward(&input).unwrap();
    let pred = output.argmax(1).unwrap();
    println!("output tensor: `{:?}`", output);
    println!("pred: `{}`", pred);
}
