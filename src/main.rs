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
    let tokenizer_bytes = utils::get_tokenizer_bytes();
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
