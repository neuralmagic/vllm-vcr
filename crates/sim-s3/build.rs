fn main() {
    println!("cargo::rustc-check-cfg=cfg(ossl300)");
    if let Ok(version) = std::env::var("DEP_OPENSSL_VERSION_NUMBER")
        && let Ok(number) = u64::from_str_radix(&version, 16)
        && number >= 0x3000_0000
    {
        println!("cargo::rustc-cfg=ossl300");
    }
}
