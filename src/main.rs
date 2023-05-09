use hound::{WavReader, SampleFormat};
use tract_onnx::{prelude::*, tract_hir::{internal::DimLike, tract_ndarray::Array}};

fn main() -> TractResult<()> {
    let window_size_samples = ( 16000. * 0.05 ) as usize;
    let model = onnx()
        .model_for_path("./silero-vad/files/silero_vad.onnx")?
        .with_input_names(["input", "h0", "c0"])?
        .with_output_names(["output", "hn", "cn"])?
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), tvec!(1, window_size_samples)),
        )?
        .with_input_fact(1, InferenceFact::dt_shape(f32::datum_type(), tvec!(2, 1, 64)))?
        .with_input_fact(2, InferenceFact::dt_shape(f32::datum_type(), tvec!(2, 1, 64)))?
        .into_optimized()?
        .into_runnable()?;
    let wav = read_wav_file("./test.wav").unwrap();
    let wav = Array::from_shape_vec((1, wav.len()), wav).unwrap();
    let wav = wav.into_arc_tensor();
    let samples = wav.shape()[1];
    let mut h = Tensor::zero::<f32>(&[2, 1, 64])?;
    let mut c = Tensor::zero::<f32>(&[2, 1, 64])?;
    let mut output: Vec<f32> = vec![];

    for ix in 0..samples.divceil(window_size_samples) {
        let offset = ix * window_size_samples;
        let mut x = Tensor::zero::<f32>(&[1, window_size_samples])?;
        let chunk_len = (samples - offset).min(window_size_samples);
        x.assign_slice(0..chunk_len, &wav, offset..offset + chunk_len, 1)?;
        let mut outputs = model.run(tvec!(x, h, c))?;
        c = outputs.remove(2).into_tensor();
        h = outputs.remove(1).into_tensor();
        output.push(outputs[0].as_slice::<f32>()?[1]);
    }

    let min_silence_duration_ms = 100;
    let min_speech_duration_ms = 250;
    let threshold = 0.5;
    let neg_threshold = 0.35;
    let min_silence_samples = min_silence_duration_ms * 16000 / 1000;
    let min_speech_samples = min_speech_duration_ms * 16000 / 1000;

    let mut triggered = false;
    let mut current_speech = 0;
    let mut temp_end = 0;

    for (ix, speech_prob) in output.into_iter().enumerate() {
        if speech_prob >= threshold && temp_end != 0 {
            temp_end = 0;
        }
        if speech_prob >= threshold && !triggered {
            triggered = true;
            current_speech = window_size_samples * ix;
        } else if speech_prob < neg_threshold && triggered {
            if temp_end == 0 {
                temp_end = window_size_samples * ix;
            }
            if (window_size_samples * ix) - temp_end >= min_silence_samples {
                if temp_end - current_speech > min_speech_samples {
                    println!("{} {}", current_speech as f32 / 16000., temp_end as f32 / 16000.);
                }
                temp_end = 0;
                triggered = false
            }
        }
    }

    Ok(())
}

fn read_wav_file(file_path: &str) -> Result<Vec<f32>, hound::Error> {
    let reader = WavReader::open(file_path)?;

    let sample_format = reader.spec().sample_format;
    let num_channels = reader.spec().channels as usize;

    let samples: Vec<_> = match sample_format {
        SampleFormat::Float => {
            reader.into_samples::<f32>().collect::<Result<Vec<_>, _>>()?
        }
        SampleFormat::Int => {
            reader
                .into_samples::<i16>()
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .map(|&sample| sample as f32 / i16::MAX as f32)
                .collect()
        }
    };

    // 将样本数据根据通道数重新排列
    let num_samples = samples.len() / num_channels;
    let mut interleaved_samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        for j in 0..num_channels {
            let index = i * num_channels + j;
            interleaved_samples.push(samples[index]);
        }
    }

    Ok(interleaved_samples)
}
