use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use hannoy::distances::Cosine;
use hannoy::{Database, Reader, Result, Writer};
use heed::EnvOpenOptions;
use ordered_float::OrderedFloat;
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use roaring::RoaringBitmap;
use tempfile::env::temp_dir;

fn main() -> Result<()> {
    let temp_dir = temp_dir();
    let env = unsafe {
        EnvOpenOptions::new()
            .map_size(1024 * 1024 * 1024 * 2) // 2GiB
            .open(temp_dir)
    }
    .unwrap();

    let dim = 768;
    let n = 500;

    let mut wtxn = env.write_txn().unwrap();
    let db: Database<Cosine> = env.create_database(&mut wtxn, None).unwrap();
    let writer: Writer<Cosine> = Writer::new(db, 0, dim);

    // generate some data & insert to hnsw
    for (item_id, vec) in load_vectors(19 * n, 0) {
        writer.add_item(&mut wtxn, item_id, &vec)?;
    }

    // build hnsw
    let mut rng = StdRng::seed_from_u64(42);
    let mut builder = writer.builder(&mut rng);
    builder.ef_construction(128);

    let now = Instant::now();
    builder.build::<16, 32>(&mut wtxn)?;
    println!("build: {:?}", now.elapsed());
    wtxn.commit()?;

    // add a few more with offsets
    let mut wtxn = env.write_txn().unwrap();
    for (item_id, vec) in load_vectors(n, 19 * n) {
        // we were tryna reload some stuff
        writer.add_item(&mut wtxn, item_id, &vec)?;
    }
    let now = Instant::now();
    builder.build::<16, 32>(&mut wtxn)?;
    println!("build: {:?}", now.elapsed());
    wtxn.commit()?;

    // search hnsw
    let data = load_vectors(20 * n, 0);
    let (_qid, query) = data[thread_rng().r#gen::<usize>() % data.len()].clone();
    let rtxn = env.read_txn()?;
    let reader = Reader::<Cosine>::open(&rtxn, 0, db).unwrap();

    let now = Instant::now();
    let nns = reader.nns(10, 10).by_vector(&rtxn, &query)?;
    println!("search: {:?}", now.elapsed());

    // check some recall
    fn l2_norm(vec: &[f32]) -> f32 {
        vec.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    let query_norm = l2_norm(&query);
    let mut opt: Vec<_> = data
        .into_iter()
        // .map(|(i, v)| {
        //     let dist: f32 = v.iter().zip(query.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        //     (dist, i as u32)
        // })
        .map(|(i, v)| {
            let dot: f32 = v.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
            let denom = l2_norm(&v) * query_norm;
            let cosine_sim = dot / denom.max(1e-6); // avoid division by zero
            (i, 0.5 - 0.5 * cosine_sim)
        })
        .collect();

    opt.sort_by_key(|(_, d)| OrderedFloat(*d));

    // println!("{:?}", &opt[..nns.len()]);
    // println!("{:?}", &nns);

    let nearest = RoaringBitmap::from_iter(opt.iter().take(nns.len()).map(|(i, _)| *i));
    let retrieved = RoaringBitmap::from_iter(nns.iter().map(|(i, _)| *i));

    println!("recall: {}", ((nearest & retrieved).len() as f64) / (nns.len() as f64));

    Ok(())
}

fn load_vectors(n: usize, offset: usize) -> Vec<(u32, Vec<f32>)> {
    let file = File::open("./assets/vectors.txt").unwrap();
    let reader = BufReader::new(&file);

    let it = reader.lines().filter_map(|line| {
        if line.is_ok() {
            let line = line.unwrap();

            if !line.starts_with("===") {
                let (id, vector) = line.split_once(',').expect(&line);
                let id: u32 = id.parse().ok()?;
                let vector: Vec<f32> = vector
                    .trim_matches(|c: char| c.is_whitespace() || c == '[' || c == ']')
                    .split(',')
                    .map(|s| s.trim().parse::<f32>().unwrap())
                    .collect();

                return Some((id, vector));
            }
        }
        None
    });

    it.skip(offset).take(n).collect()
}
