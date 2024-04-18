#![no_main]
sp1_zkvm::entrypoint!(main);

use serde::{Deserialize, Serialize};
use std::hint::black_box;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct MyPointUnaligned {
    pub x: f64,
    pub y: f64,
    pub b: bool,
}

pub fn main() {
    // let p1 = sp1_zkvm::io::read::<MyPointUnaligned>();
    // println!("Read point: {:?}", p1);
    //
    // let p2 = sp1_zkvm::io::read::<MyPointUnaligned>();
    // println!("Read point: {:?}", p2);
    let p1 = black_box(MyPointUnaligned {
        x: black_box(1.5),
        y: black_box(2.5),
        b: black_box(true),
    });

    let p2 = black_box(MyPointUnaligned {
        x: black_box(0.5),
        y: black_box(0.5),
        b: black_box(false),
    });

    let p3: MyPointUnaligned = MyPointUnaligned {
        x: p1.x + p2.x,
        y: p1.y + p2.y,
        b: p1.b && p2.b,
    };
    println!("1 Addition of 2 points: {:?}", p3);
    // sp1_zkvm::io::commit(&p3);
    let p4: MyPointUnaligned = MyPointUnaligned {
        x: p1.x + p3.x,
        y: p1.y + p3.y,
        b: p1.b && p3.b,
    };

    println!("2 Addition of 2 points: {:?}", p4);
    println!("3 END!");
}
