use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use serde_json::Value;
use rand::seq::SliceRandom;
use ndarray::{Array2, Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use itertools::Itertools;

fn save_weights_as_csv(matrix: &Array2<f32>, path: &str) {
    let file = File::create(path).expect("Unable to create CSV file");
    let mut writer = BufWriter::new(file);

    for row in matrix.genrows() {
        let line = row.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        writeln!(writer, "{}", line).expect("Unable to write row");
    }

    println!("Saved weights to {}", path);
}

fn softmax(x: &Array1<f32>) -> Array1<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    Array1::from(exps.iter().map(|&v| v / sum).collect::<Vec<_>>())
}

fn main() {
    // Load corpus from JSON
    let file = File::open("src/wikipedia_corpus.json").expect("Cannot open file");
    let reader = BufReader::new(file);
    let corpus_json: Value = serde_json::from_reader(reader).expect("Cannot parse JSON");

    let corpus: Vec<Vec<String>> = corpus_json
        .as_array()
        .expect("Expected array")
        .iter()
        .map(|line| {
            line.as_str()
                .unwrap()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect()
        })
        .collect();

    println!("Loaded {} sentences", corpus.len());

    // Build vocab
    let vocab: Vec<String> = corpus.iter().flatten().cloned().unique().sorted().collect();
    let vocab_size = vocab.len();
    let word_to_idx = vocab.iter().enumerate().map(|(i, w)| (w.clone(), i)).collect::<std::collections::HashMap<_, _>>();
    let idx_to_word = vocab.clone();

    // Generate skip-gram pairs
    let window_size = 2;
    let mut pairs = vec![];
    for sentence in &corpus {
        for i in 0..sentence.len() {
            let center = &sentence[i];
            for j in (i.saturating_sub(window_size))..=(i + window_size).min(sentence.len() - 1) {
                if i != j {
                    pairs.push((center.clone(), sentence[j].clone()));
                }
            }
        }
    }

    println!("Generated {} training pairs", pairs.len());

    // Initialize W1, W2
    let embedding_dim = 25;
    let mut w1: Array2<f32> = Array2::random((vocab_size, embedding_dim), Uniform::new(-0.01, 0.01));
    let mut w2: Array2<f32> = Array2::random((embedding_dim, vocab_size), Uniform::new(-0.01, 0.01));

    // Training
    let epochs = 1;
    let lr = 0.05;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut counter: i32 = 0;
        for (center, context) in &pairs {
            counter += 1;
            //print every 100 pairs
            if counter % 1000 == 0 {
                println!("Epoch {epoch} loss: {total_loss:.4} counter: {counter}");
            }
            let center_idx = *word_to_idx.get(center).unwrap();
            let context_idx = *word_to_idx.get(context).unwrap();

            let x = Array1::from_iter((0..vocab_size).map(|i| if i == center_idx { 1.0 } else { 0.0 }));

            let h = w1.t().dot(&x);
            let u = w2.t().dot(&h);
            let y_pred = softmax(&u);

            let mut y_true = Array1::<f32>::zeros(vocab_size);
            y_true[context_idx] = 1.0;

            let loss = -y_pred[context_idx].ln();
            total_loss += loss;

            // Backprop
            let error = &y_pred - &y_true;
            let dw2 = h.view().insert_axis(ndarray::Axis(1)).dot(&error.view().insert_axis(ndarray::Axis(0)));
            let dw1 = x.view().insert_axis(ndarray::Axis(1)).dot(&w2.dot(&error).view().insert_axis(ndarray::Axis(0)));

            w1 = &w1 - &(lr * dw1);
            w2 = &w2 - &(lr * dw2);
        }
        println!("Epoch {epoch} loss: {total_loss:.4}");
    }

    // Save w1
    save_weights_as_csv(&w1, "w1.csv");
    println!("Saved embeddings to w1.csv");
}
