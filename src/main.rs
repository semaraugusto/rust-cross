#![no_main]
sp1_zkvm::entrypoint!(main);

// use std::str::FromStr;

use anyhow::{Error as E, Result};
use candle_transformers::models::stable_lm::{Config, Model as StableLM};

use candle::{DType, Device, Tensor};
// use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

mod token_output_stream;
use token_output_stream::TokenOutputStream;

mod utils;

enum Model {
    StableLM(StableLM),
}

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

// #[cfg(not(target_os = "zkvm"))]
// // pub fn load_model(config: Config, device: &Device) -> StableLM {
// pub fn load_model(device: &Device) -> Result<LinearModel> {
//     let filenames = ["linear.safetensors"];
//     let dtype = DType::F32;
//     let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device).unwrap() };
//     Ok(Model::new(vb).unwrap())
// }

// #[cfg(target_os = "zkvm")]
// #[cfg(not(target_os="zkvm"))]
// pub fn load_model() -> Vec<u8> {
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
//         tensors.insert(tensor_name, tensor.clone());
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

// #[cfg(not(target_os = "zkvm"))]
// pub fn load_model() -> Vec<u8> {
#[cfg(target_os = "zkvm")]
pub fn load_input(device: &Device) -> Tensor {
    // pub fn load_model_2(config: Config, device: &Device) -> Result<StableLM> {
    // let dtype = DType::F32;

    // let starting_input_addr = 0x10000000usize;
    let starting_input_addr = 0xA_0000_0000usize;
    let mut addr = starting_input_addr;
    let shape_0 = utils::read_numeric::<u32>(addr);
    println!("[HERE] shape_0:: `{:?}`", shape_0);
    addr += std::mem::size_of::<u32>();
    let shape_1 = utils::read_numeric::<u32>(addr);
    println!("[HERE] shape_1:: `{:?}`", shape_1);
    let tensor_byte_len = shape_0 as usize * shape_1 as usize;
    println!("[HERE] tensor_byte_len:: `{:?}`", tensor_byte_len);
    addr += std::mem::size_of::<u32>();

    let tensor_ptr = addr as *mut u8;
    let tensor_bytes =
        // unsafe { std::slice::from_raw_parts(addr as *const u8, tensor_byte_len as usize) };
        unsafe { Vec::from_raw_parts(tensor_ptr, tensor_byte_len, tensor_byte_len) };
    addr += tensor_byte_len; // NOTE: NOT NEEDED.
    println!("tensor starting bytes: `{:?}`", &tensor_bytes[..12]);
    println!(
        "tensor end bytes: `{:?}`",
        &tensor_bytes[tensor_bytes.len() - 12..]
    );
    println!("tensor byte_len bytes: `{:?}`", tensor_byte_len);

    let mut normalized = Vec::with_capacity(tensor_byte_len);
    for (_, byte) in tensor_bytes.iter().enumerate() {
        let val = *byte as f32 / 255.0;
        // println!("HERE normalized val {}", val);
        normalized.push(val);
        // normalized[i] = (*byte as f32 / 255.0);
    }

    let magic = utils::read_numeric::<u32>(addr);
    println!("[HERE] INPUT:: `{:?}`", magic);
    assert_eq!(magic, 0x67676D6D);
    addr += std::mem::size_of::<u32>();
    println!("done normalizing {:?}", &normalized[0..784]);

    // Tensor::from_raw_buffer(&tensor_bytes, DType::U8, &[shape_0, shape_1], device).unwrap()
    // Tensor::from_raw_buffer(&tensor_bytes, DType::U8, &[1, tensor_byte_len as usize], device).unwrap()
    Tensor::from_vec(normalized.clone(), &[1, tensor_byte_len as usize], device).unwrap()
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };
        // let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = match &mut self.model {
                Model::StableLM(m) => m.forward(&input, start_pos)?,
                // Model::Quantized(m) => m.forward(&input, start_pos)?,
            };
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        // let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!("\n{generated_tokens} tokens generated",);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Which {
    V1Orig,
    V1,
    V1Zephyr,
    V2,
    V2Zephyr,
    Code,
}

#[derive(Debug)]
struct ModelParams {
    use_flash_attn: bool,

    prompt: String,

    /// The temperature used to generate samples.
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    seed: u64,

    /// The length of the sample to generate (in tokens).
    sample_len: usize,

    which: Which,

    // tokenizer_file: Option<String>,
    weight_files: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    repeat_last_n: usize,
}

#[cfg(target_os = "zkvm")]
// #[cfg(not(target_os="zkvm"))]
pub fn get_tokenizer_bytes() -> &'static [u8] {
    println!("Starting tokenizer loading");
    // let mut tokenizer_addr = 0x10000000usize;
    let mut tokenizer_addr = 0xA_2000_0000usize;
    let magic = utils::read_numeric::<u32>(tokenizer_addr);
    tokenizer_addr += std::mem::size_of::<u32>();
    println!("[HERE] VALUE:: `{:?}`", magic);
    assert_eq!(magic, 0x67676D6C);
    let tokenizer_len = utils::read_numeric::<u32>(tokenizer_addr);
    tokenizer_addr += std::mem::size_of::<u32>();
    println!("[HERE] VALUE+4:: `{:?}`", tokenizer_len);

    let tokenizer_bytes = unsafe {
        std::slice::from_raw_parts((tokenizer_addr) as *const u8, tokenizer_len as usize)
    };
    println!("tokenizer? `{:?}`", &tokenizer_bytes[0..12]);
    tokenizer_bytes
}

// fn main() -> Result<()> {
fn main() {
    // let args = Args::parse();
    let args = ModelParams { use_flash_attn: false, prompt: "What is the most efficient programming language in use? please just explain instead of providing links".to_string(), temperature: None, top_p: None, seed: 299792458, sample_len: 150, which: Which::V1Orig, weight_files: Some("/home/semar/.cache/huggingface/hub/models--stabilityai--stablelm-3b-4e1t/snapshots/fa4a6a92fca83c3b4223a3c9bf792887090ebfba/model.safetensors".to_string()), repeat_penalty: 1.1, repeat_last_n: 64 };
    println!("{:?}", args);
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => panic!("No model weights file provided"),
    };

    println!("model file path: {:?}", filenames);
    let tokenizer_bytes = get_tokenizer_bytes();
    println!("tokenizer_byte file path: {:?}", tokenizer_bytes.len());
    let tokenizer = Tokenizer::from_bytes(tokenizer_bytes).unwrap();
    println!("[SUCCESS] Tokenizer has been init!!!!!`{:?}`", tokenizer);

    // println!("tokenizer str len: {:?}", tokenizer_bytes.len());
    // // println!("tokenizer str lines[91670..91690]: {:?}", lines);
    // // println!("tokenizer json: {:?}", tokenizer_json);
    // println!("tokenizer str: {:?}", &tokenizer_bytes[1956970..1956980]);
    // // if tokenizer_str.len() > 1956980 {
    // //     println!("tokenizer str: {:?}", &tokenizer_bytes[1956970..1956980]);
    // // }
    //
    // // let tokenizer = Tokenizer::from_str(tokenizer_str).unwrap();
    //
    //
    // let config = match args.which {
    //     Which::V1Orig => Config::stablelm_3b_4e1t(args.use_flash_attn),
    //     Which::V1 | Which::V1Zephyr | Which::V2 | Which::V2Zephyr | Which::Code => {
    //         panic!("not implemented")
    //     }
    // };
    // println!("config file path: {:?}", config);
    //
    // let tokenizer = Tokenizer::from_bytes(tokenizer_bytes).unwrap();
    // println!("tokenizer has been init");
    //
    // let device = Device::Cpu;
    // let dtype = DType::F32;
    // let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device).unwrap() };
    // let model = Model::StableLM(StableLM::new(&config, vb).unwrap());
    //
    // let mut pipeline = TextGeneration::new(
    //     model,
    //     tokenizer,
    //     args.seed,
    //     args.temperature,
    //     args.top_p,
    //     args.repeat_penalty,
    //     args.repeat_last_n,
    //     &device,
    // );
    // pipeline.run(&args.prompt, args.sample_len).unwrap();
}
