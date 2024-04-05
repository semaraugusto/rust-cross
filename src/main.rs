#![no_main]
// sp1_zkvm::entrypoint!(main);
#![feature(restricted_std)]
// mod entry;

// fn float_math(x: f64, y: f64) -> f64 {
//     x + y * y * 2.0
// }
//
// fn do_string_stuff(x: &str, y: &str) -> String {
//     String::from(x) + y
// }

// fn do_vec_stuff() -> f32 {
//     let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//     v.iter().sum()
// }

// fn main(_: i32, _: *const *const u8) -> () {
//
#[no_mangle]
fn main() {
    println!("Hello, world!");
    // println!("The result of float_math is {}", float_math(1.0, 2.0));
    // println!(
    //     "The result of do_string_stuff is {}",
    //     do_string_stuff("Hello ", "World!!")
    // );
    // println!("The result of do_vec_stuff is {}", do_vec_stuff());
    // Ok(0)
}
#[no_mangle]
unsafe extern "C" fn __start() -> ! {
    // This definition of __start differs from risc0_zkvm::guest in that it does not initialize the
    // journal and will halt with empty output. It also assumes main follows the standard C
    // convention, and uses the returned i32 value as the user exit code for halt.
    let exit_code = {
        extern "C" {
            fn main(argc: i32, argv: *const *const u8) -> i32;
        }

        main(0, core::ptr::null())
    };

    loop {
        println!("Halted with exit code {}", exit_code);
    }
}

static STACK_TOP: u32 = 0x0020_0400;
// static STACK_TOP: u32 = 0x90000000;
core::arch::global_asm!(
    r#"
.section .text._start
.globl _start
_start:
    .option push;
    .option norelax
    la gp, __global_pointer$
    .option pop
    la sp, {0}
    lw sp, 0(sp)
    call __start;
"#,
    sym STACK_TOP
);
