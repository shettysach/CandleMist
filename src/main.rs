use cfg_if::cfg_if;

use api::ws;
pub mod api;
pub mod model;

#[cfg(feature = "ssr")]
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "debug");
    env_logger::init();

    use actix_files::Files;
    use actix_web::*;
    use candlemist::app::*;
    use leptos::*;
    use leptos_actix::{generate_route_list, LeptosRoutes};

    let conf = get_configuration(None).await.unwrap();
    let addr = conf.leptos_options.site_addr;
    let routes = generate_route_list(|| view! { <App/> });

    #[get("/style.css")]
    async fn css() -> impl Responder {
        actix_files::NamedFile::open_async("./style/output.css").await
    }

    let (model, tokenizer) = model_loader().unwrap();

    let mdl = web::Data::new(model);
    let tkn = web::Data::new(tokenizer);
    let dvc = web::Data::new(Device::Cpu);

    HttpServer::new(move || {
        let leptos_options = &conf.leptos_options;
        let site_root = &leptos_options.site_root;

        App::new()
            .app_data(mdl.clone())
            .app_data(tkn.clone())
            .app_data(dvc.clone())
            .service(css)
            .route("/ws", web::get().to(ws))
            .route("/api/{tail:.*}", leptos_actix::handle_server_fns())
            .leptos_routes(
                leptos_options.to_owned(),
                routes.to_owned(),
                || view! { <App/> },
            )
            .service(Files::new("/", site_root))
    })
    .bind(&addr)?
    .run()
    .await
}

cfg_if! {
    if #[cfg(feature = "ssr")] {
        use anyhow::Error as E;
        use candle_core::{quantized::gguf_file, Device};
        use candle_transformers::models::quantized_llama::ModelWeights;
        use dotenv::{dotenv, var};
        use tokenizers::Tokenizer;

        pub fn model_loader() -> Result<(ModelWeights, Tokenizer), E> {
            dotenv().ok();
            let model_path = var("MODEL_PATH").expect("MODEL_PATH is not set");
            let mut file = std::fs::File::open(&model_path)?;
            let device = Device::Cpu;
            let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;

            let mut total_size_in_bytes = 0;
            for (_, tensor) in model.tensor_infos.iter() {
                let elem_count = tensor.shape.elem_count();
                total_size_in_bytes +=
                    elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
            }
            println!(
                "\n> Loading {:?} tensors - {}",
                model.tensor_infos.len(),
                &format_size(total_size_in_bytes),
            );
            let model = ModelWeights::from_gguf(model, &mut file, &device)?;
            println!("> Successfully loaded tensors ✓");

            let tokenizer_filename =
                std::path::PathBuf::from(var("TOKENIZER_PATH").expect("TOKENIZER_PATH is not set"));
            println!("\n> Loading tokenizer");
            let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
            println!("> Successfully loaded tokenizer ✓\n");

            Ok((model, tokenizer))
        }

        fn format_size(size_in_bytes: usize) -> String {
            if size_in_bytes < 1_000 {
                format!("{}B", size_in_bytes)
            } else if size_in_bytes < 1_000_000 {
                format!("{:.2}KB", size_in_bytes as f64 / 1e3)
            } else if size_in_bytes < 1_000_000_000 {
                format!("{:.2}MB", size_in_bytes as f64 / 1e6)
            } else {
                format!("{:.2}GB", size_in_bytes as f64 / 1e9)
            }
        }
    }
}

#[cfg(not(any(feature = "ssr", feature = "csr")))]
pub fn main() {}

#[cfg(all(not(feature = "ssr"), feature = "csr"))]
pub fn main() {
    use candlemist::app::*;
    console_error_panic_hook::set_once();
    leptos::mount_to_body(App);
}
