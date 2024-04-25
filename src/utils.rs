pub trait FromBytes {
    fn from_le_bytes(a: &mut &[u8]) -> Self;
    fn from_be_bytes(a: &mut &[u8]) -> Self;
}


// #[cfg(target_os="zkvm")]
#[cfg(not(target_os="zkvm"))]
pub fn get_tokenizer_bytes() -> &'static [u8] {
    include_bytes!("/home/semar/.cache/huggingface/hub/models--stabilityai--stablelm-3b-4e1t/snapshots/fa4a6a92fca83c3b4223a3c9bf792887090ebfba/tokenizer.json")
}
#[cfg(target_os="zkvm")]
// #[cfg(not(target_os="zkvm"))]
pub fn get_tokenizer_bytes() -> &'static [u8] {
    let tokenizer_addr = 0x10000000usize;
    let magic = read_numeric::<u32>(tokenizer_addr);
    println!("[HERE] VALUE:: `{:?}`", magic);
    assert_eq!(magic, 0x67676D6C);
    let tokenizer_len = read_numeric::<u32>(tokenizer_addr+4);
    println!("[HERE] VALUE+4:: `{:?}`", tokenizer_len);

    let tokenizer_bytes = unsafe { std::slice::from_raw_parts((tokenizer_addr+8) as *const u8, tokenizer_len as usize) };
    println!("tokenizer? `{:?}`", &tokenizer_bytes[0..12]);
    tokenizer_bytes
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

pub fn read_numeric<T: FromBytes>(addr: usize) -> T {
    let mut raw_bytes = unsafe { std::slice::from_raw_parts(addr as *const u8, 8usize) };
    T::from_le_bytes(&mut raw_bytes)
}
