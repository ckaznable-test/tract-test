use hound::{WavReader, SampleFormat, WavWriter};
use rubato::{SincFixedIn, WindowFunction, InterpolationType, InterpolationParameters, Resampler};
use tract_onnx::{prelude::*, tract_hir::{internal::DimLike, tract_ndarray::Array}};

fn main() -> TractResult<()> {
    let window_size_samples = ( 16000. * 0.03 ) as usize;
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
    let src = read_wav_file("./input/input.wav").unwrap();
    let r_src = resample_audio(&src, 22050., 16000.);
    let wav = Array::from_shape_vec((1, r_src.len()), r_src).unwrap();
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

    let min_silence_duration_ms = 1000;
    let min_speech_duration_ms = 1000;
    let threshold = 0.5;
    let neg_threshold = 0.1;
    let min_silence_samples = min_silence_duration_ms * 16000 / 1000;
    let min_speech_samples = min_speech_duration_ms * 16000 / 1000;

    println!("{} {}", min_silence_samples, min_speech_samples);

    let mut triggered = false;
    let mut current_speech = 0;
    let mut temp_end = 0;

    let mut start_prob = 0.0;
    let mut file_s = 1;

    for (ix, speech_prob) in output.into_iter().enumerate() {
        if speech_prob >= threshold && temp_end != 0 {
            temp_end = 0;
        }
        if speech_prob >= threshold && !triggered {
            println!("----------------------");
            triggered = true;
            current_speech = window_size_samples * ix;
            start_prob = speech_prob;
        } else if speech_prob < neg_threshold && triggered {
            if temp_end == 0 {
                temp_end = window_size_samples * ix;
            }
            if (window_size_samples * ix) - temp_end >= min_silence_samples {
                if temp_end - current_speech > min_speech_samples {
                    println!("[{} {}] {} {}", start_prob, speech_prob, current_speech as f32 / 16000., temp_end as f32 / 16000.);
                    if let Some(data) = extract_audio_segment(&src, 22050, current_speech as f32 / 16000., temp_end as f32 / 16000.) {
                        let _ = write_wav_file(format!("output/output{}.wav", file_s).as_str(), &data, 22050, 1);
                        file_s += 1;
                    }
                }
                temp_end = 0;
                triggered = false
            }
        }
        println!("{speech_prob}");
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

fn resample_audio(input: &[f32], input_sample_rate: f64, output_sample_rate: f64) -> Vec<f32> {
    let params = InterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: InterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        output_sample_rate / input_sample_rate,
        2.0,
        params,
        input.len(),
        1,
    ).unwrap();

    let waves_in = vec![input.to_vec()];
    let out = resampler.process(&waves_in, None).unwrap();
    out[0].clone()
}

fn write_wav_file(file_path: &str, audio_data: &[f32], sample_rate: u32, num_channels: u16) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: num_channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(file_path, spec)?;

    for &sample in audio_data {
        writer.write_sample(sample)?;
    }

    writer.finalize()?;
    Ok(())
}

fn extract_audio_segment(samples: &[f32], sample_rate: u32, start_time: f32, end_time: f32) -> Option<Vec<f32>> {
    let start_sample = (start_time * sample_rate as f32) as usize;
    let mut end_sample = (end_time * sample_rate as f32) as usize;

    if start_sample >= samples.len() || end_sample >= samples.len() {
        end_sample = samples.len() - 1;
    }

    let segment = samples[start_sample..=end_sample].to_vec();
    Some(segment)
}
