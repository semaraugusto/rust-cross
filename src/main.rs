#![no_main]
sp1_zkvm::entrypoint!(main);

// use serde::{Deserialize, Serialize};
// use std::hint::black_box;
use candle_core::{Device, Tensor};

// fn main() -> Result<(), Box<dyn std::error::Error>> {
fn main() {
    let device = Device::Cpu;

    // let a = Tensor::new(&[1., 2., 3., 4., 5.], &device).unwrap();
    // let b = Tensor::new(&[1., 2., 3., 4., 5.], &device).unwrap();
    let a = Tensor::new(&[[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]], &device).unwrap();
    let b = Tensor::new(&[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]], &device).unwrap();

    let c = a.matmul(&b).unwrap();
    // let c = a.mul(&b).unwrap();

    println!("{c:?}");
    println!("`{c}`");
    // Ok(())
}
