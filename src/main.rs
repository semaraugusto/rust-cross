// #![feature(restricted_std)]

fn float_math(x: f64, y: f64) -> f64 {
    x + y * y * 2.0
}

fn do_string_stuff(x: &str, y: &str) -> String {
    String::from(x) + y
}

fn do_vec_stuff() -> f32 {
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    v.iter().sum()
}

fn main() {
    println!("Hello, world!");
    println!("The result of float_math is {}", float_math(1.0, 2.0));
    println!(
        "The result of do_string_stuff is {}",
        do_string_stuff("Hello ", "World!!")
    );
    println!("The result of do_vec_stuff is {}", do_vec_stuff());
}
