#![feature(test)]
extern crate test;
extern crate rand;
use test::Bencher;
use rand::{thread_rng, Rng};
use rand::distributions::{Distribution, Uniform};
use compressed_sparse_fiber::CompressedSparseFiber;

fn inp(max: usize) -> (Vec<u32>, f32) {
    let mut rng = rand::thread_rng();
    let die = Uniform::from(1..100);
    let vector = die.sample_iter(&mut rng).take(max).collect();
    let value  = rng.gen_range(0.0..10.0);
    (vector, value)
}

fn sample_csf(width: usize, depth: usize) -> CompressedSparseFiber<f32, u32> {
    (0..depth)
        .map(|_| inp(width))
        .into_iter()
        .collect()
}

macro_rules! make_bench {
    ($name:ident, $count:expr) => {
        mod $name {
            #[bench]
            fn expand_row(b: &mut super::Bencher) {
                let csf = super::sample_csf(8, $count);
                b.iter(|| {
                     for x in (0..10) {
                        test::black_box(csf.expand_row(x));
                    }
                });
            }
            #[bench]
            fn sum_column(b: &mut super::Bencher) {
                let csf = super::sample_csf(8, $count);
                b.iter(|| {
                    test::black_box(csf.sum_column(1));
                });
            }
        }
    };
}

make_bench!(b03_thousand, 1_000);
make_bench!(b04_hundredthousand, 100_000);
make_bench!(b05_million, 1_000_000);