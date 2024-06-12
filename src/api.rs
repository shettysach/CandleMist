use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(feature = "ssr")] {
        use std::sync::Arc;

        use candle_transformers::models::quantized_llama::ModelWeights;
        use candle_core::Device;
        use tokenizers::Tokenizer;
        mod textgen;
        use textgen::*;

        use actix_web::{web, web::Payload, Error, HttpRequest, HttpResponse};
        use actix_ws::Message as Msg;
        use futures::stream::{StreamExt};

        pub async fn ws(
            req: HttpRequest,
            body: Payload,
            model: web::Data<ModelWeights>,
            tokenizer: web::Data<Tokenizer>,
            device: web::Data<Device>
        ) -> Result<HttpResponse, Error> {
            let (response, session, mut msg_stream) = actix_ws::handle(&req, body)?;

            use std::sync::Mutex;
            use tokio::sync::mpsc;

            let (send_inference, mut recieve_inference) = mpsc::channel::<String>(100);

            let mdl: ModelWeights = model.get_ref().clone();
            let tkn: Tokenizer = tokenizer.get_ref().clone();
            let dvc: Device = device.get_ref().clone();

            let sess = Arc::new(Mutex::new(session));
            let sess_cloned = sess.clone();

            actix_rt::spawn(async move {
                let (send_new_user_message, recieve_new_user_message) =
                    std::sync::mpsc::channel::<String>();

                std::thread::spawn(move || {
                    let mut pipeline = TextGeneration::new(
                        mdl,
                        tkn,
                        fastrand::u64(..),
                        Some(0.25),
                        None,
                        None,
                        1.1,
                        64,
                        &dvc,
                    );

                    for new_user_message in recieve_new_user_message {
                        let _ = pipeline.infer(&new_user_message.to_string(), 150, send_inference.clone());
                    }
                });

                while let Some(Ok(msg)) = msg_stream.next().await {
                    match msg {
                        Msg::Ping(bytes) => {
                            let res = sess_cloned.lock().unwrap().pong(&bytes).await;
                            if res.is_err() {
                                return;
                            }
                        }
                        Msg::Text(s) => {
                            let _ = send_new_user_message.send(s.to_string());
                        }
                        _ => break,
                    }
                }
            });

            actix_rt::spawn(async move {
                while let Some(message) = recieve_inference.recv().await {
                    sess.lock().unwrap().text(message).await.expect("issue sending on websocket");
                }
                // let _ = sess.lock().unwrap().close(None).await;
            });

            Ok(response)
        }
    }
}
