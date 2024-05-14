use std::collections::HashMap;

use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_lm::{Config, Model as StableLM};

pub trait FromBytes {
    fn from_le_bytes(a: &mut &[u8]) -> Self;
    fn from_be_bytes(a: &mut &[u8]) -> Self;
}

// #[cfg(target_os="zkvm")]
#[cfg(not(target_os = "zkvm"))]
pub fn get_tokenizer_bytes() -> &'static [u8] {
    include_bytes!("/home/semar/.cache/huggingface/hub/models--stabilityai--stablelm-3b-4e1t/snapshots/fa4a6a92fca83c3b4223a3c9bf792887090ebfba/tokenizer.json")
}
// #[cfg(target_os = "zkvm")]
// // #[cfg(not(target_os="zkvm"))]
// pub fn get_tokenizer_bytes() -> &'static [u8] {
//     println!("Starting tokenizer loading");
//     // let mut tokenizer_addr = 0x10000000usize;
//     let mut tokenizer_addr = 0xA_2000_0000usize;
//     let magic = read_numeric::<u32>(tokenizer_addr);
//     tokenizer_addr += std::mem::size_of::<u32>();
//     println!("[HERE] VALUE:: `{:?}`", magic);
//     assert_eq!(magic, 0x67676D6C);
//     let tokenizer_len = read_numeric::<u32>(tokenizer_addr);
//     tokenizer_addr += std::mem::size_of::<u32>();
//     println!("[HERE] VALUE+4:: `{:?}`", tokenizer_len);
//
//     let tokenizer_bytes = unsafe {
//         std::slice::from_raw_parts((tokenizer_addr) as *const u8, tokenizer_len as usize)
//     };
//     println!("tokenizer? `{:?}`", &tokenizer_bytes[0..12]);
//     tokenizer_bytes
// }

#[cfg(not(target_os = "zkvm"))]
// pub fn load_model(config: Config, device: &Device) -> StableLM {
pub fn load_model(config: Config, device: &Device) -> Result<StableLM> {
    let filenames = ["/home/semar/.cache/huggingface/hub/models--stabilityai--stablelm-3b-4e1t/snapshots/fa4a6a92fca83c3b4223a3c9bf792887090ebfba/model.safetensors"];
    // let filenames = ["here_0.safetensors", "here_1.safetensors", "here_2.safetensors", "here_3.safetensors"];
    // let device = Device::Cpu;
    let dtype = DType::F32;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device).unwrap() };
    // println!("vb: `{:?}`", vb);
    // let model_data = unsafe { Vec::from_raw_parts(model_ptr, model_len as usize, model_len as usize) };
    Ok(StableLM::new(&config, vb).unwrap())
}

#[cfg(target_os = "zkvm")]
// #[cfg(not(target_os="zkvm"))]
// pub fn load_model() -> Vec<u8> {
pub fn load_model(config: Config, device: &Device) -> Result<StableLM> {
    // pub fn load_model_2(config: Config, device: &Device) -> Result<StableLM> {
    let dtype = DType::F32;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    let starting_model_addr = 0x10000000usize;
    let mut addr = starting_model_addr;
    let magic = read_numeric::<u32>(addr);
    addr += std::mem::size_of::<u32>();
    println!("[HERE] VALUE:: `{:?}`", magic);
    assert_eq!(magic, 0x67676D6C);
    // let model_len = read_numeric::<u64>(addr);
    // println!("[HERE] VALUE+4:: `{:?}`", model_len);
    // let model_addr = model_addr + 4;
    // let model_ptr = model_addr as *mut u8;
    let num_tensors = read_numeric::<u32>(addr);
    addr += std::mem::size_of::<u32>();
    println!("[HERE] num_tensors:: `{:?}`", num_tensors);
    for _ in 0..num_tensors {
        let string_len = read_numeric::<u32>(addr);
        addr += std::mem::size_of::<u32>();
        println!("[HERE] string_len: `{:?}`", string_len);
        let raw_bytes =
            unsafe { std::slice::from_raw_parts(addr as *const u8, string_len as usize) };
        addr += string_len as usize;
        let tensor_name = String::from_utf8_lossy(raw_bytes).to_string();
        println!("tensor_name: `{:?}`", tensor_name);
        println!("name_len: `{:?}`", tensor_name.len());
        // let tensor_byte_len = read_numeric::<u32>(addr);
        // addr += std::mem::size_of::<u32>();
        // let tensor_len = tensor_byte_len / 4;
        // println!("tensor_len: `{:?}`", tensor_len);
        let tensor_shape_0 = read_numeric::<u32>(addr);
        addr += std::mem::size_of::<u32>();
        let tensor_shape_1 = read_numeric::<u32>(addr);
        addr += std::mem::size_of::<u32>();

        println!(
            "tensor_shape: (`{:?}`, `{:?}`)",
            tensor_shape_0, tensor_shape_1
        );
        let tensor_ptr = addr as *mut u8;
        println!("tensor_ptr: `{:?}`", tensor_ptr);
        let tensor_byte_len = tensor_shape_0 * tensor_shape_1 * std::mem::size_of::<f32>() as u32;
        println!("tensor_byte_len: `{:?}`", tensor_byte_len);
        let tensor_data = unsafe {
            Vec::from_raw_parts(
                tensor_ptr,
                tensor_byte_len as usize,
                tensor_byte_len as usize,
            )
        };
        addr += tensor_byte_len as usize;

        println!("tensor_data[0..12]: `{:?}`", &tensor_data[0..12]);
        let tensor = Tensor::from_raw_buffer(
            &tensor_data,
            DType::F32,
            &[tensor_shape_0 as usize, tensor_shape_1 as usize],
            device,
        )
        .unwrap();
        // DType::BF16 => convert_slice::<half::bf16>(data, shape, device),
        // DType::F16 => convert_slice::<half::f16>(data, shape, device),
        // DType::F32 => convert_slice::<f32>(data, shape, device),
        println!("tensor: `{}`", tensor);
        println!("tensor debug: `{:?}`", tensor);
        // tensors[&tensor_name] = tensor;
        tensors.insert(tensor_name, tensor);
    }
    println!("tensor keys: `{:?}`", tensors.keys());

    Err(E::msg("not implemented"))

    // let model_data = unsafe { Vec::from_raw_parts(model_ptr, model_len as usize, model_len as usize) };
    // println!("model? `{:?}`", &model_data[0..12]);
    // println!("model size? `{:?}`", model_data.len());
    // // unimplemented!();
    // let vb = VarBuilder::from_buffered_safetensors(model_data, dtype, device).unwrap();
    // Ok(StableLM::new(&config, vb).unwrap())
}

impl<const N: usize> FromBytes for [u8; N] {
    fn from_le_bytes(a: &mut &[u8]) -> [u8; N] {
        let (int_bytes, rest) = a.split_at(N);

        let mut me = [0u8; N];
        me.copy_from_slice(int_bytes);

        *a = rest;
        me
    }
    fn from_be_bytes(a: &mut &[u8]) -> [u8; N] {
        let (int_bytes, rest) = a.split_at(N);

        let mut me = [0u8; N];
        me.copy_from_slice(int_bytes);

        *a = rest;
        me
    }
}

impl FromBytes for u64 {
    fn from_le_bytes(a: &mut &[u8]) -> u64 {
        u64::from_le_bytes(FromBytes::from_le_bytes(a))
    }
    fn from_be_bytes(a: &mut &[u8]) -> u64 {
        u64::from_be_bytes(FromBytes::from_le_bytes(a))
    }
}
impl FromBytes for u32 {
    fn from_le_bytes(a: &mut &[u8]) -> u32 {
        u32::from_le_bytes(FromBytes::from_le_bytes(a))
    }
    fn from_be_bytes(a: &mut &[u8]) -> u32 {
        u32::from_be_bytes(FromBytes::from_le_bytes(a))
    }
}
impl FromBytes for u8 {
    fn from_le_bytes(a: &mut &[u8]) -> u8 {
        u8::from_le_bytes(FromBytes::from_le_bytes(a))
    }
    fn from_be_bytes(a: &mut &[u8]) -> u8 {
        u8::from_be_bytes(FromBytes::from_le_bytes(a))
    }
}

impl FromBytes for i64 {
    fn from_le_bytes(a: &mut &[u8]) -> i64 {
        i64::from_le_bytes(FromBytes::from_le_bytes(a))
    }
    fn from_be_bytes(a: &mut &[u8]) -> i64 {
        i64::from_be_bytes(FromBytes::from_le_bytes(a))
    }
}
impl FromBytes for i32 {
    fn from_le_bytes(a: &mut &[u8]) -> i32 {
        i32::from_le_bytes(FromBytes::from_le_bytes(a))
    }
    fn from_be_bytes(a: &mut &[u8]) -> i32 {
        i32::from_be_bytes(FromBytes::from_le_bytes(a))
    }
}

impl FromBytes for usize {
    fn from_le_bytes(a: &mut &[u8]) -> usize {
        usize::from_le_bytes(FromBytes::from_le_bytes(a))
    }
    fn from_be_bytes(a: &mut &[u8]) -> usize {
        usize::from_be_bytes(FromBytes::from_le_bytes(a))
    }
}
impl FromBytes for isize {
    fn from_le_bytes(a: &mut &[u8]) -> isize {
        isize::from_le_bytes(FromBytes::from_le_bytes(a))
    }
    fn from_be_bytes(a: &mut &[u8]) -> isize {
        isize::from_be_bytes(FromBytes::from_le_bytes(a))
    }
}
// pub fn load_input(addr: &mut usize) -> Tensor {}

// pub fn read_string(mut addr: usize) -> String {
pub fn read_string(addr: &mut usize) -> String {
    let string_len = read_numeric::<u32>(*addr);
    *addr += std::mem::size_of::<u32>();
    println!("[HERE] string_len: `{:?}`", string_len);
    let raw_bytes = unsafe { std::slice::from_raw_parts(*addr as *const u8, string_len as usize) };
    // addr = addr + string_len as usize;
    *addr += string_len as usize;
    String::from_utf8_lossy(raw_bytes).to_string()
}

// pub fn read_numeric<T: FromBytes>(addr: usize) -> (T, usize) {
pub fn read_numeric<T: FromBytes>(addr: usize) -> T {
    // let mut raw_bytes = unsafe { std::slice::from_raw_parts(addr as *const u8, 8usize) };
    let mut raw_bytes =
        unsafe { std::slice::from_raw_parts(addr as *const u8, std::mem::size_of::<T>()) };
    // (T::from_le_bytes(&mut raw_bytes), addr)
    T::from_le_bytes(&mut raw_bytes)
}
