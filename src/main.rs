#![no_main]
sp1_zkvm::entrypoint!(main);

use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module};

struct Model {
    first: Linear,
    second: Linear,
}

impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(image)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }
}

fn main() {
    let device = Device::Cpu;

    let weight = Tensor::randn(0f32, 1.0, (100, 784), &device).unwrap();
    let bias = Tensor::randn(0f32, 1.0, (100, ), &device).unwrap();
    let first = Linear::new(weight, Some(bias));
    let weight = Tensor::randn(0f32, 1.0, (10, 100), &device).unwrap();
    let bias = Tensor::randn(0f32, 1.0, (10, ), &device).unwrap();
    let second = Linear::new(weight, Some(bias));
    let model = Model { first, second };

    let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device).unwrap();

    let digit = model.forward(&dummy_image).unwrap();
    println!("Digit {digit:?} digit");
    println!("Digit {digit} digit");
    // Ok(())
}
