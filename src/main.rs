#![no_main]
sp1_zkvm::entrypoint!(main);

use anyhow::{Error as E, Result};
// use clap::{Parser, ValueEnum};

// use candle_transformers::models::quantized_stable_lm::Model as QStableLM;
use candle_transformers::models::stable_lm::{Config, Model as StableLM};

use candle::{DType, Device, Tensor};
// use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
// use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

mod token_output_stream;
use token_output_stream::TokenOutputStream;

enum Model {
    StableLM(StableLM),
}

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    // tokenizer: Tokenizer,
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
            // tokenizer,
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
        println!(
            "\n{generated_tokens} tokens generated",
        );
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

// #[derive(Parser, Debug)]
// #[command(author, version, about, long_about = None)]
// struct Args {
//     /// Run on CPU rather than on GPU.
//     #[arg(long)]
//     cpu: bool,
//
//     /// Enable tracing (generates a trace-timestamp.json file).
//     #[arg(long)]
//     tracing: bool,
//
//     #[arg(long)]
//     use_flash_attn: bool,
//
//     #[arg(long)]
//     prompt: String,
//
//     /// The temperature used to generate samples.
//     #[arg(long)]
//     temperature: Option<f64>,
//
//     /// Nucleus sampling probability cutoff.
//     #[arg(long)]
//     top_p: Option<f64>,
//
//     /// The seed to use when generating random samples.
//     #[arg(long, default_value_t = 299792458)]
//     seed: u64,
//
//     /// The length of the sample to generate (in tokens).
//     #[arg(long, short = 'n', default_value_t = 1000)]
//     sample_len: usize,
//
//     #[arg(long)]
//     model_id: Option<String>,
//
//     #[arg(long, default_value = "main")]
//     revision: String,
//
//     #[arg(long, default_value = "v2")]
//     which: Which,
//
//     #[arg(long)]
//     tokenizer_file: Option<String>,
//
//     #[arg(long)]
//     weight_files: Option<String>,
//
//     #[arg(long)]
//     quantized: bool,
//
//     /// Penalty to be applied for repeating tokens, 1. means no penalty.
//     #[arg(long, default_value_t = 1.1)]
//     repeat_penalty: f32,
//
//     /// The context size to consider for the repeat penalty.
//     #[arg(long, default_value_t = 64)]
//     repeat_last_n: usize,
// }

#[derive(Debug)]
struct ModelParams {
    /// Run on CPU rather than on GPU.
    cpu: bool,

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

    model_id: Option<String>,

    revision: String,

    which: Which,

    // tokenizer_file: Option<String>,

    weight_files: Option<String>,

    quantized: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    repeat_last_n: usize,
}

// fn main() -> Result<()> {
fn main() {
    // let args = Args::parse();
    let args = ModelParams { cpu: false, use_flash_attn: false, prompt: "What is the most efficient programming language in use? please just explain instead of providing links".to_string(), temperature: None, top_p: None, seed: 299792458, sample_len: 150, model_id: None, revision: "main".to_string(), which: Which::V1Orig, weight_files: Some("/home/semar/.cache/huggingface/hub/models--stabilityai--stablelm-3b-4e1t/snapshots/fa4a6a92fca83c3b4223a3c9bf792887090ebfba/model.safetensors".to_string()), quantized: false, repeat_penalty: 1.1, repeat_last_n: 64 };
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

    // let start = std::time::Instant::now();
    // let api = Api::new()?;
    // let model_id = match args.model_id {
    //     Some(model_id) => model_id,
    //     None => match args.which {
    //         Which::V1Orig => "lmz/candle-stablelm-3b-4e1t".to_string(),
    //         Which::V1 => "stabilityai/stablelm-3b-4e1t".to_string(),
    //         Which::V1Zephyr => "stabilityai/stablelm-zephyr-3b".to_string(),
    //         Which::Code => "stabilityai/stable-code-3b".to_string(),
    //         Which::V2 => "stabilityai/stablelm-2-1_6b".to_string(),
    //         Which::V2Zephyr => "stabilityai/stablelm-2-zephyr-1_6b".to_string(),
    //     },
    // };

    // let repo = api.repo(Repo::with_revision(
    //     model_id,
    //     RepoType::Model,
    //     args.revision,
    // ));
    // let tokenizer_filename = match args.tokenizer_file {
    //     Some(file) => std::path::PathBuf::from(file),
    //     None => panic!("No tokenizer file provided"),
    // };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => panic!("No model weights file provided"),
    };

    // println!("retrieved the files in {:?}", start.elapsed());
    println!("model file path: {:?}", filenames);
    // let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    // tokenizer_filename.t
    // let tokenizer = Tokenizer::from_bytes(include_bytes!("/home/semar/.cache/huggingface/hub/models--stabilityai--stablelm-3b-4e1t/snapshots/fa4a6a92fca83c3b4223a3c9bf792887090ebfba/tokenizer.json")).map_err(E::msg)?;
    let tokenizer = Tokenizer::from_bytes(include_bytes!("/home/semar/.cache/huggingface/hub/models--stabilityai--stablelm-3b-4e1t/snapshots/fa4a6a92fca83c3b4223a3c9bf792887090ebfba/tokenizer.json")).unwrap();
    println!("tokenizer has been init");

    // let start = std::time::Instant::now();
    let config = match args.which {
        Which::V1Orig => Config::stablelm_3b_4e1t(args.use_flash_attn),
        Which::V1 | Which::V1Zephyr | Which::V2 | Which::V2Zephyr | Which::Code => {
            panic!("not implemented")
            // let config_filename = repo.get("config.json")?;
            // let config = std::fs::read_to_string(config_filename)?;
            // let mut config: Config = serde_json::from_str(&config)?;
            // config.set_use_flash_attn(args.use_flash_attn);
            // config
        }
    };
    println!("config file path: {:?}", config);

    // let device = candle_examples::device(args.cpu)?;
    let device = Device::Cpu;
    let (model, device) = {
        // let dtype = if device.is_cuda() {
        //     DType::BF16
        // } else {
        //     DType::F32
        // };
        let dtype = DType::F32;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device).unwrap() };
        let model = StableLM::new(&config, vb).unwrap();
        (Model::StableLM(model), device)
    };

    // println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    pipeline.run(&args.prompt, args.sample_len).unwrap();
    // Ok(())
}
