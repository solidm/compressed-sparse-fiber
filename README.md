compressed-sparse-fiber
=========

[![Build Status](https://travis-ci.com/solidm/compressed-sparse-fiber.svg?branch=master)](https://travis-ci.com/rust-lang/compressed-sparse-fiber)
[![Crates.io](https://img.shields.io/crates/v/compressed-sparse-fiber.svg)](https://crates.io/crates/compressed-sparse-fiber)
[![Documentation](https://docs.rs/compressed-sparse-fiber/badge.svg)](https://docs.rs/compressed-sparse-fiber)

CSF is a generalization of compressed sparse row (CSR) index.
See [smith2017knl](http://shaden.io/pub-files/smith2017knl.pdf)

CSF index recursively compresses each dimension of a tensor into a set
of prefix trees. Each path from a root to leaf forms one tensor
non-zero index. CSF is implemented with two arrays of buffers and one
arrays of integers.

### Example usage

```rust
let rows = vec![
    (vec![1, 1, 1, 2], 1.0),
    (vec![1, 1, 1, 3], 2.0),
    (vec![1, 2, 1, 1], 3.0),
    (vec![1, 2, 1, 3], 4.0),
    (vec![1, 2, 2, 1], 5.0),
    (vec![2, 2, 2, 1], 6.0),
    (vec![2, 2, 2, 2], 7.0),
    (vec![2, 2, 2, 3], 8.0),
];

let csf: CompressedSparseFiber<_,_> = rows.into_iter().collect();                       
```

The above could be represented by a structure like this:
```
CompressedSparseFiber { 
    fptr: [[0, 2, 3], [0, 1, 3, 4], [0, 2, 4, 5, 8]], 
    fids: [[1, 2], [1, 2, 2], [1, 1, 2, 2], [2, 3, 1, 3, 1, 1, 2, 3]], 
    vals: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] 
}
```

### Retreiving in uncompressed form 

```rust
for t in csf {
    println!("{:?}", t);
}
```

### References
 - [Apache Arrow](https://github.com/apache/arrow/blob/master/format/SparseTensor.fbs)
 - [smith2017knl](http://shaden.io/pub-files/smith2017knl.pdf) 
 - [Tensor-Matrix Products with a Compressed Sparse Tensor](https://www.researchgate.net/publication/283457552_Tensor-Matrix_Products_with_a_Compressed_Sparse_Tensor)

