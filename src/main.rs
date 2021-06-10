use concrete::*;
use std::time::{Duration, Instant};
use itertools::izip;
use std::io::prelude::*;
use std::fs::File; use std::io::{BufReader}; 
use ndarray::{Array1, Array2, Array3, arr1, arr2, arr3}; 

fn mul_ctxt_message_example(folder:&str){
    println!("\n\n=======================================");
    println!("mul_ctxt_message_example start...");
    use itertools::izip;
    // encoder
    let encoder = Encoder::new(-2., 6., 4, 4).unwrap();

    let(rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1) = load_keys(folder);
// two lists of messages
    let messages_1: Vec<f64> = vec![-1., 2., 0., 5., -0.5];
    let messages_2: Vec<f64> = vec![-2., -1., 3., 2.5, 1.5];

    // encode and encrypt
    let mut ciphertext =
        VectorRLWE::encode_encrypt(&rlwe_key, &messages_1, &encoder).unwrap();

    // multiplication between ciphertext and messages_2
    let max_constant: f64 = 3.;
    let scalar_precision: usize = 4;
    ciphertext
        .mul_constant_with_padding_inplace(&messages_2, max_constant, scalar_precision)
        .unwrap();

    // decryption
    let (decryptions, dec_encoders) = ciphertext.decrypt_with_encoders(&rlwe_key).unwrap();

    // check the precision loss related to the encryption
    for (before_1, before_2, after, enc) in izip!(
        messages_1.iter(),
        messages_2.iter(),
        decryptions.iter(),
        dec_encoders.iter()
    ) {
        if (before_1 * before_2 - after).abs() > enc.get_granularity() / 2. {
            panic!();
        }
    }

   print!("\n");
   print!("messages1: ");
   for m in messages_1.iter(){
     print!("{:.3}, ", m);
   }
   print!("\n");

   print!("messages2: ");
   for m in messages_2.iter(){
     print!("{:.3}, ", m);
   }
   print!("\n");

   print!("dec     : ");
   for m in decryptions.iter(){
     print!("{:.3}, ", m);
   }
   print!("\n");

    println!("mul_ctxt_message_example done...");

}


fn raw_relu(x:f64) -> f64{
    if(x > 0.){
        return x;
    }else{
        return 0.;
    }
}

fn raw_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

fn bs_max_example(folder:&str){
    println!("\n\n=======================================");
    println!("bs_sigmoid_example start...");
    // encoders
    let ecd1 = Encoder::new(-10., 10., 8, 4).unwrap();
    let ecd2 = Encoder::new(-10., 10., 8, 4).unwrap();
    let ecd_out = Encoder::new(-10., 10., 8, 4).unwrap();

    let(rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1) = load_keys(folder);

    // messages
    let message1: Vec<f64> = vec![-5.];
    let message2: Vec<f64> = vec![2.];

    // encode and encrypt
    let mut c1 = VectorLWE::encode_encrypt(&lwe_key1, &message1, &ecd1).unwrap();
    let mut c2 = VectorLWE::encode_encrypt(&lwe_key1, &message2, &ecd2).unwrap();

    let c3 = c1.sub_with_padding(&c2).unwrap();
    println!("debug1");

    // bootstrap
    let from_index:usize = 0;
    let c4 = c3.bootstrap_nth_with_function(&bsk, raw_relu, &ecd_out, from_index).unwrap();
    println!("debug2");

    let c5 = c2.add_with_padding(&c4).unwrap();
    println!("debug3");

    // decrypt
    let d5 = c5.decrypt_decode(&lwe_key2).unwrap();
    println!("debug4");

    print!("max function of (message1[{}] = {}, message2[{}] = {})\n", from_index, message1[from_index], from_index, message2[from_index]);
    print!(" = {}\n", d5[0]);

    println!("bs_max_example done.. ");

}



fn bs_sigmoid_example(folder:&str){
    println!("\n\n=======================================");
    println!("bs_sigmoid_example start...");
    // encoders
    let encoder_input = Encoder::new(-10., 10., 4, 1).unwrap();
    let encoder_output = Encoder::new(-1., 1., 4, 0).unwrap();

    let(rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1) = load_keys(folder);

    // messages
    let message: Vec<f64> = vec![-5.];

    // encode and encrypt
    let start = Instant::now();
    let ciphertext_input =
        VectorLWE::encode_encrypt(&lwe_key1, &message, &encoder_input).unwrap();
    let end = start.elapsed();
    println!("enc done.");
    println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);

    // bootstrap
    let from_index:usize = 0;
    let start = Instant::now();
    let ciphertext_output = ciphertext_input
        .bootstrap_nth_with_function(&bsk, raw_sigmoid, &encoder_output, from_index)
        .unwrap();
    let end = start.elapsed();
    println!("bs done.");
    println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);

    // decrypt
    let start = Instant::now();
    let decryption = ciphertext_output.decrypt_decode(&lwe_key2).unwrap();
    let end = start.elapsed();
    println!("dec done.");
    println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);

    print!("sigmoid function of message[{}] = {}\n", from_index, message[from_index]);
    print!(" = {}\n", decryption[0]);

    println!("bs_sigmoid_example done.. ");

}


fn bs_relu_example(folder:&str){
    println!("\n\n=======================================");
    println!("bs_relu_example start...");
    // encoders
    let encoder_input = Encoder::new(-10., 10., 4, 1).unwrap();
    let encoder_output = Encoder::new(0., 10., 4, 0).unwrap();

    let(rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1) = load_keys(folder);

    // messages
    let message: Vec<f64> = vec![-5.];

    // encode and encrypt
    let start = Instant::now();
    let ciphertext_input =
        VectorLWE::encode_encrypt(&lwe_key1, &message, &encoder_input).unwrap();
    let end = start.elapsed();
    println!("enc done.");
    println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);

    // bootstrap
    let from_index:usize = 0;
    let start = Instant::now();
    let ciphertext_output = ciphertext_input
        .bootstrap_nth_with_function(&bsk, raw_relu, &encoder_output, from_index)
        .unwrap();
    let end = start.elapsed();
    println!("bs done.");
    println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);

    // decrypt
    let start = Instant::now();
    let decryption = ciphertext_output.decrypt_decode(&lwe_key2).unwrap();
    let end = start.elapsed();
    println!("dec done.");
    println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);

    print!("square function of message[{}] = {}\n", from_index, message[from_index]);
    print!(" = {}\n", decryption[0]);

    println!("bs_relu_example done.. ");

}

fn sub_example(folder:&str){
    println!("\n\n=======================================");
    println!("sub_example start...");
    // encoder
    let encoder = Encoder::new(100., 110., 7, 1).unwrap();


    let folder = "keys1";
    let(rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1) = load_keys(folder);

    // two lists of messages
    let messages1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
    let messages2: Vec<f64> = vec![105.276, 103.3, 100.12, 100.1, 106.78];

    // encode and encrypt
    let mut ciphertext_1 =
        VectorRLWE::encode_encrypt_packed(&rlwe_key, &messages1, &encoder).unwrap();
    let ciphertext_2 =
        VectorRLWE::encode_encrypt_packed(&rlwe_key, &messages2, &encoder).unwrap();

    // subtraction between ciphertext and messages_2
    ciphertext_1.sub_with_padding_inplace(&ciphertext_2);

    // decryption
    let decryptions: Vec<f64> = ciphertext_1.decrypt_decode(&rlwe_key).unwrap();

    // check the precision loss related to the encryption
    for (before_1, before_2, after, enc) in izip!(
        messages1.iter(),
        messages2.iter(),
        decryptions.iter(),
        ciphertext_1.encoders.iter()
    ) {
        if (before_1 - before_2 - after).abs() > enc.get_granularity() / 2. {
            panic!("decryption: {}", after);
        }
    }

   print!("\n");
   print!("messages1: ");
   for m in messages1.iter(){
     print!("{:.3}, ", m);
   }
   print!("\n");

   print!("messages2: ");
   for m in messages2.iter(){
     print!("{:.3}, ", m);
   }
   print!("\n");

   print!("dec     : ");
   for m in decryptions.iter(){
     print!("{:.3}, ", m);
   }
   print!("\n");
    println!("sub_example done...");

}

fn add_example(folder:&str){
    println!("\n\n=======================================");
    println!("add_example start...");
   // encoder
   let encoder_1 = Encoder::new(100., 110., 7, 1).unwrap();
   let encoder_2 = Encoder::new(-30., -20., 7, 1).unwrap();


   //let folder = "keys1";
   let(rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1) = load_keys(folder);
   //let secret_key = LWESecretKey::new(&LWE128_630);

   // two lists of messages
   let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
   let messages_2: Vec<f64> = vec![-22., -27.5, -21.2, -29., -25.];

   // encode and encrypt
   let mut ciphertext_1 =
       VectorLWE::encode_encrypt(&lwe_key1, &messages_1, &encoder_1).unwrap();
   let ciphertext_2 = VectorLWE::encode_encrypt(&lwe_key1, &messages_2, &encoder_2).unwrap();

   // addition between ciphertext and messages_2
   ciphertext_1.add_with_padding_inplace(&ciphertext_2);

   // decryption
   let decryptions: Vec<f64> = ciphertext_1.decrypt_decode(&lwe_key1).unwrap();

   // check the precision loss related to the encryption
   for (before_1, before_2, after, enc) in izip!(
       messages_1.iter(),
       messages_2.iter(),
       decryptions.iter(),
       ciphertext_1.encoders.iter()
   ) {
       if (before_1 - before_2 - after).abs() > enc.get_granularity() / 2. {
           panic!();
       }
   }

   print!("\n");
   print!("messages1: ");
   for m in messages_1.iter(){
     print!("{:.3}, ", m);
   }
   print!("\n");
   print!("messages2: ");
   for m in messages_2.iter(){
     print!("{:.3}, ", m);
   }
   print!("\n");

   print!("dec     : ");
   for m in decryptions.iter(){
     print!("{:.3}, ", m);
   }
   print!("\n");
   println!("sub_example done...");

}

fn ks_1_2_example(folder:&str){
    println!("\n\n=======================================");
    println!("ks_1_2_example start...");
   let encoder = Encoder::new(100., 110., 8, 0).unwrap();

    let folder = "keys1";
    let(rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1) = load_keys(folder);

   // a list of messages that we encrypt
   let messages: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
   let mut ciphertext_tmp1 =
       VectorLWE::encode_encrypt(&lwe_key1, &messages, &encoder).unwrap();

   // key switch
   let start = Instant::now();
   let ciphertext_tmp2 = ciphertext_tmp1.keyswitch(&ksk_1_2).unwrap();
   let end = start.elapsed();
   println!("ks done..");
   println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);

   // decryption
   let decryptions: Vec<f64> = ciphertext_tmp2.decrypt_decode(&lwe_key2).unwrap();

   // check the precision loss related to the encryption
   for (before, after) in messages.iter().zip(decryptions.iter()) {
       if (before - after).abs() > encoder.get_granularity() / 2. {
           panic!();
       }
   }

   print!("\n");
   print!("messages: ");
   for m in messages.iter(){
     print!("{:.3}, ", m);
   }
   print!("\n");

   print!("dec     : ");
   for m in decryptions.iter(){
     print!("{:.3}, ", m);
   }
   print!("\n");

   println!("ks_1_2_example done...");

}

fn ks_2_1_example(folder:&str){
    println!("\n\n=======================================");
    println!("ks_2_1_example start...");
   let encoder = Encoder::new(100., 110., 8, 0).unwrap();

    let(rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1) = load_keys(folder);

   // a list of messages that we encrypt
   let messages: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
   let mut ciphertext_tmp1 =
       VectorLWE::encode_encrypt(&lwe_key2, &messages, &encoder).unwrap();

   // key switch
   let start = Instant::now();
   let ciphertext_tmp2 = ciphertext_tmp1.keyswitch(&ksk_2_1).unwrap();
   let end = start.elapsed();
   println!("ks done..");
   println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);

   // decryption
   let decryptions: Vec<f64> = ciphertext_tmp2.decrypt_decode(&lwe_key1).unwrap();

   // check the precision loss related to the encryption
   for (before, after) in messages.iter().zip(decryptions.iter()) {
       if (before - after).abs() > encoder.get_granularity() / 2. {
           panic!();
       }
   }

   print!("\n");
   print!("messages: ");
   for m in messages.iter(){
     print!("{:.3}, ", m);
   }
   print!("\n");

   print!("dec     : ");
   for m in decryptions.iter(){
     print!("{:.3}, ", m);
   }
   print!("\n");

   println!("ks_2_1_example done...");

}
fn ks_ks_example(folder:&str){
    println!("\n\n=======================================");
    println!("ks_ks_example start...");
   let encoder = Encoder::new(100., 110., 8, 0).unwrap();

    let(rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1) = load_keys(folder);

   // a list of messages that we encrypt
   let messages: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
   let mut ciphertext_tmp1 =
       VectorLWE::encode_encrypt(&lwe_key1, &messages, &encoder).unwrap();

   // key switch lwe_key1 -> lwe_key2
   let start = Instant::now();
   let ciphertext_tmp2 = ciphertext_tmp1.keyswitch(&ksk_1_2).unwrap();
   let end = start.elapsed();
   println!("ks_1_2 done..");
   println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);


   // key switch lwe_key2 -> lwe_key1
   let start = Instant::now();
   let ciphertext_tmp3 = ciphertext_tmp2.keyswitch(&ksk_2_1).unwrap();
   let end = start.elapsed();
   println!("ks_2_1 done..");
   println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);

   // decryption
   let decryptions: Vec<f64> = ciphertext_tmp3.decrypt_decode(&lwe_key1).unwrap();

   // check the precision loss related to the encryption
   for (before, after) in messages.iter().zip(decryptions.iter()) {
       if (before - after).abs() > encoder.get_granularity() / 2. {
           panic!();
       }
   }
   println!("ks_ks_example done...");

}


fn generate_rlwe_key(rlwe_params: RLWEParams) -> RLWESecretKey{
    let rlwe_secret_key = RLWESecretKey::new(&rlwe_params);
    println!("generated rlwe_key");
    return rlwe_secret_key;
}

fn generate_lwe_key_lwe_key(rlwe_key: &RLWESecretKey, lwe_params: LWEParams) -> (LWESecretKey, LWESecretKey){
    let lwe_key1 = LWESecretKey::new(&lwe_params);
    let lwe_key2 = rlwe_key.to_lwe_secret_key();
    println!("generated lwe_key1, lwe_key2");
    return (lwe_key1, lwe_key2);
}

fn generate_ksk(k1: &LWESecretKey, k2: &LWESecretKey ) -> (LWEKSK, LWEKSK){
    let ksk_1_2 = LWEKSK::new(&k1, &k2, 8, 3);
    let ksk_2_1 = LWEKSK::new(&k2, &k1, 8, 3);
    println!("generated ksk");
    return (ksk_1_2, ksk_2_1);
}

fn generate_bsk(rlwe_key: &RLWESecretKey, lwe_key1: &LWESecretKey, beta: usize, level: usize) -> LWEBSK{
    let bsk = LWEBSK::new(&lwe_key1, &rlwe_key, beta, level);
    println!("generated bsk");
    return bsk;
}


fn generate_keys( lwe_params: LWEParams, rlwe_params: RLWEParams,beta:usize, level:usize) -> (RLWESecretKey, LWESecretKey, LWESecretKey, LWEBSK, LWEKSK, LWEKSK){
    let rlwe_key = generate_rlwe_key(rlwe_params);
    let (lwe_key1, lwe_key2) = generate_lwe_key_lwe_key(&rlwe_key, lwe_params);
    let bsk = generate_bsk(&rlwe_key, &lwe_key1, beta, level);
    let (ksk_1_2, ksk_2_1) = generate_ksk(&lwe_key1, &lwe_key2);
    return (rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1);
}

fn generate_and_save_keys(folder: &str, lwe_params: LWEParams, rlwe_params: RLWEParams, beta: usize, level: usize){
    //let folder_path = folder.to_string();
    let (rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1) = generate_keys(lwe_params, rlwe_params, beta, level);
    let mut tmp_path = folder.to_string() + "/rlwe_key.json";
    rlwe_key.save(&tmp_path);
    tmp_path = folder.to_string() + "/lwe_key1.json";
    lwe_key1.save(&tmp_path);
    tmp_path = folder.to_string() + "/lwe_key2.json";
    lwe_key2.save(&tmp_path);
    tmp_path = folder.to_string() + "/bsk.json";
    bsk.save(&tmp_path);
    tmp_path = folder.to_string() + "/ksk_1_2.json";
    ksk_1_2.save(&tmp_path);
    tmp_path = folder.to_string() + "/ksk_2_1.json";
    ksk_2_1.save(&tmp_path);
    println!("done...");
}


fn load_rlwe_key(folder: &str) -> (RLWESecretKey){
    let tmp_path = folder.to_string() + "/rlwe_key.json";
    let rlwe_key = RLWESecretKey::load(&tmp_path);
    println!("loaded rlwe_key");
    return (rlwe_key.unwrap());
}


fn load_lwe_key_lwe_key(folder: &str) -> (LWESecretKey, LWESecretKey){
    let mut tmp_path = folder.to_string() + "/lwe_key1.json";
    let lwe_key1 = LWESecretKey::load(&tmp_path);
    tmp_path = folder.to_string() + "/lwe_key2.json";
    let lwe_key2 = LWESecretKey::load(&tmp_path);
    println!("loaded lwe_key1, lwe_key2");
    return (lwe_key1.unwrap(), lwe_key2.unwrap());

}

fn load_ksk(folder: &str) -> (LWEKSK, LWEKSK) {
    let start = Instant::now();
    let mut tmp_path = folder.to_string() + "/ksk_1_2.json";
    let ksk_1_2 = LWEKSK::load(&tmp_path);
    tmp_path = folder.to_string() + "/ksk_2_1.json";
    let ksk_2_1 = LWEKSK::load(&tmp_path);
    let end = start.elapsed();
    println!("loaded ksk");
    println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);
    return (ksk_1_2, ksk_2_1);
}


fn load_bsk(folder: &str) -> (LWEBSK){

    let start = Instant::now();
    let tmp_path = folder.to_string() + "/bsk.json";
    let bootstrapping_key = LWEBSK::load(&tmp_path);
    let end = start.elapsed();
    println!("loaded bsk");
    println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);

    return (bootstrapping_key);
}



fn load_keys(folder: &str) -> (RLWESecretKey, LWESecretKey, LWESecretKey, LWEBSK, LWEKSK, LWEKSK){
    let rlwe_key = load_rlwe_key(folder);
    let (lwe_key1, lwe_key2) = load_lwe_key_lwe_key(folder);
    let bsk = load_bsk(folder);
    let (ksk_1_2, ksk_2_1) = load_ksk(folder);

    return (rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1);

}



fn bs_example(folder:&str){
    println!("\n\n=======================================");
    println!("bs_example start...");
    // encoders
    let start = Instant::now();
    let encoder_input = Encoder::new(-10., 10., 6, 1).unwrap();

    let folder = "keys1";
    let(rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1) = load_keys(folder);

    // messages
    let message: Vec<f64> = vec![-5.];

    // encode and encrypt
    let ciphertext_input =
        VectorLWE::encode_encrypt(&lwe_key1, &message, &encoder_input).unwrap();

    let from_index:usize = 0;
    let start = Instant::now();
    // bootstrap
    let ciphertext_output = ciphertext_input
        .bootstrap_nth(&bsk, from_index)
        .unwrap();

    let end = start.elapsed();
    println!("bs done");
    println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);
    // decrypt
    let decryption = ciphertext_output.decrypt_decode(&lwe_key2).unwrap();
    println!("decryption: {}", decryption[0]);

    print!("\n");
    print!("bs of message[{}] = {}\n", from_index, message[from_index]);
    print!(" = {}", decryption[0]);

    println!("bs_example done..");
}

fn bs_ks_example(folder:&str){
    println!("\n\n=======================================");
    println!("bs_ks_example start...");
    // encoders
    let start = Instant::now();
    let encoder_input = Encoder::new(-10., 10., 6, 1).unwrap();

    let(rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1) = load_keys(folder);

    // messages
    let message: Vec<f64> = vec![-5.];

    // encode and encrypt
    let ciphertext_input =
        VectorLWE::encode_encrypt(&lwe_key1, &message, &encoder_input).unwrap();

    let start = Instant::now();
    // bootstrap
    let ciphertext_output = ciphertext_input
        .bootstrap_nth(&bsk, 0)
        .unwrap();

    let end = start.elapsed();
    println!("bs done");
    println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);

    // key switch
    let start = Instant::now();
    let ciphertext_ksk = ciphertext_output.keyswitch(&ksk_2_1).unwrap();
    let end = start.elapsed();
    println!("ks done..");
    println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);

    // decrypt
    let decryption = ciphertext_ksk.decrypt_decode(&lwe_key1).unwrap();
    println!("decryption: {}", decryption[0]);


    println!("after ks(bs(Enc([-5]))): {}", decryption[0]);

    println!("bs_ks_example done..");
}

fn bs_square_example(folder:&str){
    println!("\n\n=======================================");
    println!("bs_square_example start...");
    // encoders
    let encoder_input = Encoder::new(-10., 10., 4, 1).unwrap();
    let encoder_output = Encoder::new(0., 100., 4, 0).unwrap();

    let folder = "keys1";
    let(rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1) = load_keys(folder);

    // messages
    let message: Vec<f64> = vec![-5.];

    // encode and encrypt
    let start = Instant::now();
    let ciphertext_input =
        VectorLWE::encode_encrypt(&lwe_key1, &message, &encoder_input).unwrap();
    let end = start.elapsed();
    println!("enc done.");
    println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);

    // bootstrap
    let from_index:usize = 0;
    let start = Instant::now();
    let ciphertext_output = ciphertext_input
        .bootstrap_nth_with_function(&bsk, |x| x * x, &encoder_output, from_index)
        .unwrap();
    let end = start.elapsed();
    println!("bs done.");
    println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);

    // decrypt
    let start = Instant::now();
    let decryption = ciphertext_output.decrypt_decode(&lwe_key2).unwrap();
    let end = start.elapsed();
    println!("dec done.");
    println!("{}.{:03} sec passed", end.as_secs(), end.subsec_nanos() / 1_000_000);

    if (decryption[0] - message[0] * message[0]).abs() > encoder_output.get_granularity() {
        panic!(
            "decryption: {} / expected value: {} / granularity: {}",
            decryption[0],
            message[0] * message[0],
            encoder_output.get_granularity()
        );
    }

    print!("square function of message[{}] = {}\n", from_index, message[from_index]);
    print!(" = {}", decryption[0]);

    println!("bs_square_example done.. ");

}


fn mult_ctxt_ctxt_example(folder:&str){
    println!("\n\n=======================================");
    println!("mult_ctxt_ctxt_example start...");
    // encoders
    let encoder_1 = Encoder::new(10., 20., 6, 2).unwrap();
    let encoder_2 = Encoder::new(-30., -20., 6, 2).unwrap();

    let folder = "keys1";
    let (rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1) = load_keys(folder);

    // two lists of messages
    let messages_1: Vec<f64> = vec![10.276, 14.3, 14.12, 11.1, 17.78];
    let messages_2: Vec<f64> = vec![-22., -27.5, -21.2, -29., -25.];

    // encode and encrypt
    let ciphertext_1 =
        VectorLWE::encode_encrypt(&lwe_key1, &messages_1, &encoder_1).unwrap();
    let ciphertext_2 =
        VectorLWE::encode_encrypt(&lwe_key1, &messages_2, &encoder_2).unwrap();

    let from_1:usize = 0;
    let from_2:usize = 1;
    // multiplication
    let ciphertext_res = ciphertext_1
        .mul_from_bootstrap_nth(&ciphertext_2, &bsk, from_1, from_2)
        .unwrap();

    // decrypt
    let decryption = ciphertext_res.decrypt_decode(&lwe_key2).unwrap();

    print!("\n");
    print!("mult of message_1[{}] = {}, message_2[{}] = {}\n", from_1, messages_1[from_1], from_2, messages_2[from_2]);
    print!(" = {}\n", decryption[0]);


    println!("mult_ctxt_ctxt_example done...");
}

fn str2vec64(filename:&str) -> Vec<f64>{
    print!("str2vec64 called\n");
    let file = File::open(filename).expect("Unable to open file");
    let reader = BufReader::new(file);
    let mut val: Vec<f64> =  Vec::new();
    for line in reader.lines() {
        val.push(line.unwrap().parse::<f64>().unwrap());
    }
    println!("{:?}", val);
    return val;
}

fn line_to_num(line_str: &str) -> Vec<f64> {
    // let mut result1 = Vec::new();
    // println!("Input str = {}", line_str);
    let nums = line_str
        .trim()
        .split(' ')
        .flat_map(str::parse::<f64>)
        .collect::<Vec<_>>();
    //for num in &nums {
    //    println!("{}", num);
    //}
    return nums;
}

fn lines_to_num(filepath: &str) -> Vec<Vec<f64>>{
    let file = File::open(filepath).expect("Unable to open file");
    let reader = BufReader::new(file);
    let mut val: Vec<Vec<f64>> = Vec::new();
    for line in reader.lines() {
        // val.push(line.unwrap().parse::<f64>().unwrap());
        let line_str = line.unwrap();
        // println!("Line string= {:?}", line_str);
        val.push(line_to_num(&line_str));
    }
    //println!("{:?}", val);
    return val;
  }

fn vec_to_ndarray_1(x: &Vec<f64>) -> Array1::<f64> {
  let mut xa = Array1::<f64>::zeros(x.len());
  //println!("xa: {:?}", xa);
  for (i, el) in x.iter().enumerate(){
    xa[i] = x[i];
  }
  return xa;

}


fn lr_example(folder:&str){
  println!("hello, logistic_regression");
  let encoder = Encoder::new(-20., 20., 8, 4).unwrap();

  println!("debug-2");
  let w = lines_to_num("w.txt");
  let b = lines_to_num("b.txt");
  let x = lines_to_num("x.txt");

  println!("debug-1");
  let wa = vec_to_ndarray_1(&w[0]);
  let ba = vec_to_ndarray_1(&b[0]);
  let xa = vec_to_ndarray_1(&x[0]);
  
  println!("wa: {:?}", wa);
  println!("ba: {:?}", ba);
  println!("xa: {:?}", xa);

  println!("debug-0");
  let tmp1 = wa.dot(&xa);
  let tmp2 = tmp1 + ba;
  let tmp3 = raw_sigmoid(tmp2[0]);
  println!("{:?}", tmp1);
  println!("{:?}", tmp2);
  println!("{:?}", tmp3);
  println!("debug0");


  let(rlwe_key, lwe_key1, lwe_key2, bsk, ksk_1_2, ksk_2_1) = load_keys(folder);

  // encode and encrypt
  let mut c1 = VectorRLWE::encode_encrypt(&rlwe_key, &x[0], &encoder).unwrap();

  // multiplication between ciphertext and messages_2
  let max_constant: f64 = 5.;
  let scalar_precision: usize = 4;
  c1.mul_constant_with_padding_inplace(&w[0], max_constant, scalar_precision).unwrap();

  println!("debug1");
  //let lwe_ct = c1.extract_1_lwe(0, 0).unwrap();
  //// decrypt
  //println!("debug2");
  //let decryption = lwe_ct.decrypt_decode(&lwe_key1).unwrap();
  //println!("decryption of lwe: {}", decryption[0]);
  //c1.add_with_padding_inplace(&b[0]);

  // decryption
  let (decryptions, dec_encoders) = c1.decrypt_with_encoders(&rlwe_key).unwrap();

  for d in decryptions.iter(){
    print!("{}, ", d);
  }
  println!();
}


fn extract_example(){
    // encoder
    let encoder = Encoder::new(-10., 10., 8, 2).unwrap();

    // generate a fresh secret key
    let secret_key = RLWESecretKey::new(&RLWE128_1024_1);

    // a list of messages
    let messages: Vec<f64> = vec![-6.276, 4.3, 0.12, -1.1, 7.78];

    // encode and encrypt
    //let plaintext = Plaintext::encode(&messages, &encoder).unwrap();
    let rlwe_ct = VectorRLWE::encode_encrypt_packed(&secret_key, &messages, &encoder).unwrap();
    println!("debug0");

    // extraction of the coefficient indexed by 2 (0.12)
    // in the RLWE ciphertext indexed by 0
    let lwe_ct = rlwe_ct.extract_1_lwe(0, 0).unwrap();

    println!("Well done :-)");

} 

fn main() {
    println!("\nHello, world!\n");

    //let (lwe_key1, lwe_key2) = load_lwe_key_lwe_key();
    //let (ksk_1_2, ksk_2_1) = generate_ksk(&lwe_key1, &lwe_key2);
    //ksk_1_2.save("ksk_1_2.json");
    //ksk_2_1.save("ksk_2_1.json");


    // for generating keys and save them in ./keys1
    let mut key_folder = String::from("keys2");
    //let lwe_param = LWE80_2048;
    //let rlwe_param = RLWE80_2048_1;
    //let mut beta:usize = 4;
    //let mut level:usize = 5;
    //generate_and_save_keys(&key_folder, lwe_param, rlwe_param, beta, level);

    //key_folder = String::from("keys12");
    //beta = 4;
    //level = 6;
    //generate_and_save_keys(&key_folder, lwe_param, rlwe_param, beta, level);

    //key_folder = String::from("keys13");
    //beta = 4;
    //level = 7;
    //generate_and_save_keys(&key_folder, lwe_param, rlwe_param, beta, level);

    //key_folder = String::from("keys14");
    //beta = 4;
    //level = 8;
    //generate_and_save_keys(&key_folder, lwe_param, rlwe_param, beta, level);

    //// for generating keys and save them in ./keys1
    //key_folder = String::from("keys15");
    //beta = 3;
    //level = 8;
    //generate_and_save_keys(&key_folder, lwe_param, rlwe_param, beta, level);

    //key_folder = String::from("keys16");
    //beta = 3;
    //level = 9;
    //generate_and_save_keys(&key_folder, lwe_param, rlwe_param, beta, level);

    //key_folder = String::from("keys17");
    //beta = 3;
    //level = 10;
    //generate_and_save_keys(&key_folder, lwe_param, rlwe_param, beta, level);

    //generate_bsk(&key_folder);
    //add_example(&key_folder);
    //sub_example(&key_folder);
    //mul_ctxt_message_example(&key_folder);
    //mult_ctxt_ctxt_example(&key_folder);
    //bs_square_example(&key_folder);
    //bs_example(&key_folder);
    //ks_1_2_example(&key_folder);
    //ks_2_1_example(&key_folder);
    //ks_ks_example(&key_folder);
    //bs_ks_example(&key_folder);
    //bs_relu_example(&key_folder);
    //bs_sigmoid_example(&key_folder);

    //bs_max_example(&key_folder);
    //lr_example(&key_folder);
    extract_example();

}
