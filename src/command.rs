pub mod commmand {
    use clap::Parser;

    #[derive(Parser, Debug, Clone)]
    pub struct Args {
        /// Address to listen
        #[arg(short, long, default_value = "[::1]:50051")]
        listen: String,

        /// Model type, default text
        #[arg(short, long, default_value_t = false)]
        vision_mode: bool,

        /// Vision model input image size, default 224
        #[arg(short, long, default_value_t = 224)]
        input_image_size: usize,

        /// Whether to pad and truncate the input text token sequence to 77
        #[arg(short, long, default_value_t = false)]
        pad_token_sequence: bool,
    }

    impl Args {
        pub fn get_listen(&self) -> String {
            self.listen.clone()
        }
        pub fn get_vision_mode(&self) -> bool {
            self.vision_mode.clone()
        }
        pub fn get_input_image_size(&self) -> usize {
            self.input_image_size.clone()
        }
        pub fn get_pad_token_sequence(&self) -> bool {
            self.pad_token_sequence.clone()
        }
    }
}
