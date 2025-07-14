use std::{fs::File, io::{BufRead, BufReader}, time::Instant};

use hannoy::{distances::Cosine, Database, Reader, Result, Writer};
use heed::EnvOpenOptions;
use ordered_float::OrderedFloat;
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use roaring::RoaringBitmap;

fn main() -> Result<()> {
    let env = unsafe {
        EnvOpenOptions::new()
            .map_size(1024 * 1024 * 1024 * 2) // 2GiB
            .open("./")
    }
    .unwrap();

    let dim = 768;
    let n = 1000;

    let mut wtxn = env.write_txn().unwrap();
    let db: Database<Cosine> = env.create_database(&mut wtxn, None).unwrap();
    let writer: Writer<Cosine> = Writer::new(db, 0, dim);

    // generate some data & insert to hnsw
    for (item_id, vec) in load_vectors(n){
        writer.add_item(&mut wtxn, item_id as u32, &vec)?;
    }

    // build hnsw
    let mut rng = StdRng::seed_from_u64(42);
    let mut builder = writer.builder(&mut rng);
    builder.ef_construction(400);

    let now = Instant::now();
    builder.build(&mut wtxn)?;
    println!("build: {:?}", now.elapsed());
    wtxn.commit()?;

    // search hnsw
    let data = load_vectors(n);
    let (qid, query) = data[thread_rng().gen::<usize>()%data.len()].clone();
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
            (i as u32, 0.5 - 0.5 * cosine_sim)
        })
        .collect();

    opt.sort_by_key(|(_, d)| OrderedFloat(*d));

    // println!("{:?}", &opt[..nns.len()]);
    // println!("{:?}", &nns);

    let recall = 0;
    let nearest = RoaringBitmap::from_iter(opt.iter().take(nns.len()).map(|(i, _)| *i));
    let retrieved = RoaringBitmap::from_iter(nns.iter().map(|(i, _)| *i));

    println!("recall: {}", ((nearest & retrieved).len() as f64) / (nns.len() as f64));

    Ok(())
}

fn load_vectors(n: usize) -> Vec<(u32, Vec<f32>)>{
    
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

    return it.take(n).collect();
}
