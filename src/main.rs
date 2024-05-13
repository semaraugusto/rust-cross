#![no_main]
sp1_zkvm::entrypoint!(main);

mod utils;

use anyhow::Error as E;
// use candle::
use candle::{DType, Device, Result, Shape, Tensor, D};
use candle_nn::{loss, ops, Conv2d, Linear, Module, ModuleT, Optimizer, VarBuilder, VarMap};
use rand::prelude::*;
use std::{borrow::Borrow, collections::HashMap};

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

#[derive(Clone)]
enum WhichModel {
    Linear,
    Mlp,
}

// #[cfg(not(target_os = "zkvm"))]
// // pub fn load_model(config: Config, device: &Device) -> StableLM {
// pub fn load_model(device: &Device) -> Result<LinearModel> {
//     let filenames = ["linear.safetensors"];
//     let dtype = DType::F32;
//     let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device).unwrap() };
//     Ok(Model::new(vb).unwrap())
// }
//
// #[cfg(target_os = "zkvm")]
// // #[cfg(not(target_os="zkvm"))]
// // pub fn load_model() -> Vec<u8> {
// pub fn load_model(device: &Device) -> Result<LinearModel> {
//     // pub fn load_model_2(config: Config, device: &Device) -> Result<StableLM> {
//     let dtype = DType::F32;
//     let mut tensors: HashMap<String, Tensor> = HashMap::new();
//
//     // let starting_model_addr = 270557184usize;
//     let starting_model_addr = 0x5_0000_0000usize;
//     let mut addr = starting_model_addr;
//     let magic = utils::read_numeric::<u32>(addr);
//     addr += std::mem::size_of::<u32>();
//     println!("[HERE] VALUE:: `{:?}`", magic);
//     assert_eq!(magic, 0x67676D6C);
//     // let model_len = read_numeric::<u64>(addr);
//     // println!("[HERE] VALUE+4:: `{:?}`", model_len);
//     // let model_addr = model_addr + 4;
//     // let model_ptr = model_addr as *mut u8;
//     let num_tensors = utils::read_numeric::<u32>(addr);
//     addr += std::mem::size_of::<u32>();
//     println!("[HERE] num_tensors:: `{:?}`", num_tensors);
//     for _ in 0..num_tensors {
//         let string_len = utils::read_numeric::<u32>(addr);
//         addr += std::mem::size_of::<u32>();
//         println!("[HERE] string_len: `{:?}`", string_len);
//         let raw_bytes =
//             unsafe { std::slice::from_raw_parts(addr as *const u8, string_len as usize) };
//         addr += string_len as usize;
//         let tensor_name = String::from_utf8_lossy(raw_bytes).to_string();
//         println!("tensor_name: `{:?}`", tensor_name);
//         println!("name_len: `{:?}`", tensor_name.len());
//         let tensor_dims = utils::read_numeric::<u32>(addr);
//         addr += std::mem::size_of::<u32>();
//         // let mut tensor_shape = vec![];
//         let mut tensor_shape = vec![];
//         let mut tensor_num_elems = 1;
//         for _ in 0..tensor_dims {
//             let shape_i = utils::read_numeric::<u32>(addr);
//             tensor_num_elems *= shape_i;
//             addr += std::mem::size_of::<u32>();
//             tensor_shape.push(shape_i as usize)
//         }
//         let tensor_byte_len = tensor_num_elems * std::mem::size_of::<f32>() as u32;
//         println!("tensor_shape!: `{:?}`", tensor_shape);
//         println!("tensor_byte_len!: `{:?}`", tensor_byte_len);
//         let tensor_ptr = addr as *mut u8;
//         let tensor_bytes =
//             // unsafe { std::slice::from_raw_parts(addr as *const u8, tensor_byte_len as usize) };
//             unsafe { Vec::from_raw_parts(tensor_ptr, tensor_byte_len as usize, tensor_byte_len as usize) };
//         addr += tensor_byte_len as usize;
//         println!("tensor starting bytes: `{:?}`", &tensor_bytes[..40]);
//         println!(
//             "tensor end bytes: `{:?}`",
//             &tensor_bytes[tensor_bytes.len() - 40..]
//         );
//         let float_vec = tensor_bytes
//             .chunks_exact(std::mem::size_of::<f32>())
//             .map(|chunk| {
//                 let mut bytes = [0u8; 4];
//                 bytes.copy_from_slice(chunk);
//                 f32::from_le_bytes(bytes)
//             })
//             .collect::<Vec<f32>>();
//         // tensor_shape.reverse();
//         // let tensor =
//         //     Tensor::from_raw_buffer(&tensor_bytes, DType::F32, &tensor_shape, device).unwrap();
//         if tensor_name == "weight" {
//             // println!("tensor sum: `{:?}`", tensor.to_vec2::<f32>()?);
//             // println!("tensor bytes: `{:?}`", &tensor_bytes[..784]);
//             // println!("tensor[0]: `{:?}`", tensor.to_vec2::<f32>()?[0]);
//             println!("tensor[0]: `{:?}`", &float_vec[..784]);
//         }
//         let tensor = Tensor::from_vec(float_vec.clone(), tensor_shape, device).unwrap();
//         println!("tensor: `{}`", tensor);
//         println!("tensor debug: `{:?}`", tensor);
//         tensors.insert(tensor_name, tensor);
//     }
//     let magic = utils::read_numeric::<u32>(addr);
//     addr += std::mem::size_of::<u32>();
//     println!("[HERE] MODEL:: `{:?}`", magic);
//     assert_eq!(magic, 0x67676D6D);
//
//     println!("tensor keys: `{:?}`", tensors.keys());
//
//     let vb = VarBuilder::from_tensors(tensors, dtype, device);
//     Ok(LinearModel::new(vb).unwrap())
// }
//
// #[cfg(not(target_os = "zkvm"))]
// pub fn load_input(device: &Device) -> Tensor {
//     let m = candle_datasets::vision::mnist::load().unwrap();
//     println!("train-images: {:?}", m.train_images.shape());
//     println!("train-labels: {:?}", m.train_labels.shape());
//     println!("test-images: {:?}", m.test_images.shape());
//     println!("test-labels: {:?}", m.test_labels.shape());
//     let image = m.test_images.get(0).unwrap().unsqueeze(0).unwrap();
//     println!("image: {:?}", image.shape());
//     image
// }
//
// // #[cfg(not(target_os="zkvm"))]
// // pub fn load_model() -> Vec<u8> {
// // #[cfg(target_os = "zkvm")]
// pub fn load_input(device: &Device) -> Tensor {
//     // pub fn load_model_2(config: Config, device: &Device) -> Result<StableLM> {
//     // let dtype = DType::F32;
//
//     // let starting_input_addr = 0x10000000usize;
//     let starting_input_addr = 0xA_0000_0000usize;
//     let mut addr = starting_input_addr;
//     let shape_0 = utils::read_numeric::<u32>(addr);
//     println!("[HERE] shape_0:: `{:?}`", shape_0);
//     addr += std::mem::size_of::<u32>();
//     let shape_1 = utils::read_numeric::<u32>(addr);
//     println!("[HERE] shape_1:: `{:?}`", shape_1);
//     let tensor_byte_len = shape_0 * shape_1;
//     println!("[HERE] tensor_byte_len:: `{:?}`", tensor_byte_len);
//     addr += std::mem::size_of::<u32>();
//
//     let tensor_ptr = addr as *mut u8;
//     let tensor_bytes =
//         // unsafe { std::slice::from_raw_parts(addr as *const u8, tensor_byte_len as usize) };
//         unsafe { Vec::from_raw_parts(tensor_ptr, tensor_byte_len as usize, tensor_byte_len as usize) };
//     addr += tensor_byte_len as usize; // NOTE: NOT NEEDED.
//     println!("tensor starting bytes: `{:?}`", &tensor_bytes[..12]);
//     println!(
//         "tensor end bytes: `{:?}`",
//         &tensor_bytes[tensor_bytes.len() - 12..]
//     );
//     println!("tensor byte_len bytes: `{:?}`", tensor_byte_len);
//
//     // let mut normalized = Vec::with_capacity(tensor_byte_len); // NOTE: VEC::WITH_CAPACITY IS BUGGED.
//     let mut normalized = vec![];
//     for (_, byte) in tensor_bytes.iter().enumerate() {
//         let val = *byte as f32 / 255.0;
//         // println!("HERE normalized val {}", val);
//         normalized.push(val);
//         // normalized[i] = (*byte as f32 / 255.0);
//     }
//
//     let magic = utils::read_numeric::<u32>(addr);
//     println!("[HERE] INPUT:: `{:?}`", magic);
//     assert_eq!(magic, 0x67676D6D);
//     addr += std::mem::size_of::<u32>();
//     println!("done normalizing {:?}", &normalized[0..784]);
//
//     // Tensor::from_raw_buffer(&tensor_bytes, DType::U8, &[shape_0, shape_1], device).unwrap()
//     // Tensor::from_raw_buffer(&tensor_bytes, DType::U8, &[1, tensor_byte_len as usize], device).unwrap()
//     Tensor::from_vec(normalized, &[tensor_byte_len as usize], device).unwrap()
// }

struct Args {
    model: WhichModel,
}

// pub fn main() -> anyhow::Result<()> {
pub fn main() {
    println!("Starting");
    let device = Device::Cpu;

    println!("start loading model...");
    // let model = load_model(&device).unwrap();
    println!("start loading input...");
    // let input = load_input(&device);
    let dtype = DType::F32;
    // let size = 4; // (i<=4)                   initial vec: ok - second vec: ok    - output ok
    // let size = 5; // (i >= 5 and i <= 8)      initial vec: ok - second vec: ok    - output 0-vec
    // let size = 9; // (i >= 9 and i <= 12)     initial vec: ok - second vec: 0-vec - output 0-vec
    // let size = 13; // (i >= 13 and i <= 20)   initial vec: ok - second vec: 0-vec - output ok
    // let size = 21; // (i >= 21 and i <= 24)   initial vec: ok - second vec: 0-vec - output 0-vec
    // let size = 29; // (i >= 29 and i <= 36)    initial vec: ok - second vec: ok    - output ok
    // let size = 37; // (i >= 37 and i <= 40)    initial vec: ok - second vec: ok    - output 0-vec
    // let size = 41; // (i >= 41 and i <= 44)    initial vec: ok - second vec: 0-vec    - output 0-vec
    // let size = 45; // (i >= 45 and i <= 52)    initial vec: ok - second vec: 0-vec    - output ok
    // let size = 53; // (i >= 53 and i <= 56)    initial vec: ok - second vec: 0-vec    - output 0-vec
    // let size = 57; // (i >= 57 and i <= 60)    initial vec: ok - second vec: ok    - output 0-vec
    // let size = 61; // (i >= 61 and i <= 68)    initial vec: ok - second vec: ok    - output ok

    let size = 4;
    // let weight_tensor = Tensor::rand(0f32, 1f32, &[size], &device).unwrap();
    // println!(
    //     "weight_tensor storage: {:?}",
    //     weight_tensor.storage_and_layout()
    // );
    let mut rng = rand::thread_rng();
    let mut weight_values = Vec::with_capacity(size);
    let uniform = rand::distributions::Uniform::new(0f32, 1f32);
    for _i in 0..size {
        weight_values.push(rng.sample::<f32, _>(uniform))
    }
    println!("weight_values: {:?}", weight_values);

    let mut input_values = Vec::with_capacity(size);
    for _i in 0..size {
        input_values.push(rng.sample::<f32, _>(uniform))
    }
    println!("input_values: {:?}", input_values);

    let mut expected_output = Vec::with_capacity(size);
    for i in 0..size {
        expected_output.push(weight_values[i] + input_values[i])
    }
    println!("expected_output: {:?}", expected_output);
    // let output = weight_tensor.add(&input).unwrap();
    let weight_tensor = Tensor::from_vec(weight_values, &[size], &device).unwrap();
    println!("weight_tensor: {}", weight_tensor);
    println!("weight_tensor: {:p}", &weight_tensor);

    let input_tensor = Tensor::from_vec(input_values, &[size], &device).unwrap();
    println!("input_tensor: {}", input_tensor);
    println!("input_tensor: {:p}", &input_tensor);

    let output_tensor = weight_tensor.add(&input_tensor).unwrap();
    println!("output_tensor: {}", output_tensor);
    assert_eq!(output_tensor.to_vec1::<f32>().unwrap(), expected_output);
}
