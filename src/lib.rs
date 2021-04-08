use std::hash::Hash;
use sequence_trie::SequenceTrie;
use std::collections::hash_map::RandomState;
use std::iter::Sum;

type Row<T, U> = (Vec<U>, T);
type Rows<T, U> = Vec<Row<T, U>>;

#[derive(Debug, Clone)]
pub struct CompressedSparseFiberBuilder<T, U>
    where U: Eq + Hash + Ord {
    rows: SequenceTrie<U, T>
}

impl<T, U> CompressedSparseFiberBuilder<T, U>
    where U: Eq + Hash + Clone + Ord {
    pub fn new() -> CompressedSparseFiberBuilder<T, U> {
        CompressedSparseFiberBuilder { rows: SequenceTrie::new() }
    }

    pub fn build(self) -> CompressedSparseFiber<T, U>
        where U: Eq + Hash + Clone + Copy + Ord, T: Copy
    {
        CompressedSparseFiber::from(&self.rows)
    }
}

#[derive(Debug, Clone)]
struct IteratorState {
    next_index: usize
}

impl Default for IteratorState {
    fn default() -> IteratorState {
        IteratorState {
            next_index: 0
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompressedSparseFiber<T, U> {
    fptr: Vec<Vec<usize>>,
    fids: Vec<Vec<U>>,
    vals: Vec<T>,
    _state: IteratorState,
}

impl<'a, T: 'a, U> CompressedSparseFiber<T, U> where U: Clone {
    pub fn new(fptr: Vec<Vec<usize>>,
           fids: Vec<Vec<U>>,
           vals: Vec<T>) -> CompressedSparseFiber<T, U> {
        CompressedSparseFiber { fptr, fids, vals, _state: IteratorState { next_index: 0 } }
    }

    fn dim(self: &CompressedSparseFiber<T, U>, level: usize, index: usize) -> Option<(usize, U)> {
        for j in 0..self.fptr[level].len() {
            let v = self.fptr[level][j];
            if v > index {
                return Some((j - 1, self.fids[level][j - 1].clone()));
            }
        }
        None
    }

    fn expand_row(self: &CompressedSparseFiber<T, U>, index: usize) -> Row<T, U>
        where T: Copy, U: Copy {
        let val = self.vals[index];
        let depth = self.fids.len();

        // The last row has the same length as vals
        let mut result = vec![self.fids[depth - 1][index]];
        let mut current_index = index;
        for x in (0..depth - 1).rev() {
            match self.dim(x, current_index) {
                Some((new_index, node)) => {
                    current_index = new_index;
                    result.push(node);
                }
                None => ()
            }
        }
        result.reverse();
        (result, val)
    }

    fn weights(self: &CompressedSparseFiber<T, U>, col_index: usize) -> Vec<usize> {

        fn combine(current_weights: Vec<usize>, fptr_row: Vec<usize>) -> Vec<usize> {
            let tail = &fptr_row[1..];
            fptr_row.iter()
                .zip(tail)
                .map(|(&i,&j)| &current_weights[i..j])
                .map(|x|x.iter().sum::<usize>())
                .collect::<Vec<_>>()
        }

        let fptr_row = &self.fptr[self.fptr.len() - 1];
        let tail = &fptr_row[1..];
        let initial = fptr_row.iter()
            .zip(tail)
            .map(|(&x, &y)| y - x)
            .collect::<Vec<_>>();

        (&self.fptr[col_index..self.fptr.len() - 1])
            .iter()
            .rfold(initial, |w, f| combine(w, f.to_vec()))
    }

    pub fn sum_column(self: &CompressedSparseFiber<T, U>, col_index: usize) -> U
        where T: Copy, U: Sum<U> + Copy {
        let row = self.fids[col_index].clone();

        let a = if col_index == self.fptr.len() {
            row.into_iter().sum::<U>()
        } else {
            let w = self.weights(col_index);
            // Repeat and take to avoid Mul<usize, Output = U> constraint
            row.iter()
                .zip(w)
                .map(|(&x,y)| std::iter::repeat(x).take(y).sum())
                .sum::<U>()
        };

        a
    }
}

impl<T, U> From<&SequenceTrie<U, T>> for CompressedSparseFiber<T, U>
    where T: Copy,
          U: Clone + Eq + Hash + Ord + Copy {
   fn from(trie: &SequenceTrie<U, T, RandomState>) -> Self {
        let mut i = vec![trie];
        let mut fids: Vec<Vec<U>> = vec![];
        let mut fptr = vec![];
        let mut vals: Vec<T> = vec![];
        let mut initial = true;

        while !i.is_empty() {
            let mut offset = 0;
            let mut fptr_row = vec![0];
            let (keys, children): (Vec<&U>, Vec<_>) = i.into_iter()
                .flat_map(|y| {
                    let mut x = y.children_with_keys();
                    offset += x.len();
                    fptr_row.push(offset);
                    x.sort_by(|(a, _), (b, _)| a.cmp(b));
                    x
                })
                .unzip();
            if !keys.is_empty() {
                let row = keys.into_iter().map(|&f| f).collect();
                fids.push(row);
                if !initial {
                    fptr.push(fptr_row);
                } else {
                    initial = false;
                }
            }

            let mut values: Vec<T> = children.iter()
                .filter_map(|x| x.value())
                .map(|&x| x)
                .collect::<Vec<_>>();
            vals.append(&mut values);
            i = children
        }
        CompressedSparseFiber::new(fptr, fids, vals)
    }
}

impl<T, U> From<&Rows<T, U>> for CompressedSparseFiber<T, U>
    where T: Copy,
          U: Clone + Eq + Hash + Ord + Copy {
    fn from(rows: &Rows<T, U>) -> CompressedSparseFiber<T, U> {
        let mut trie: SequenceTrie<U, T> = SequenceTrie::new();
        for (row, x) in rows {
            trie.insert(row, *x);
        }
        CompressedSparseFiber::<T, U>::from(&trie)
    }
}

impl<T, U> Iterator for CompressedSparseFiber<T, U>
    where T: Copy, U: Clone + Copy {
    type Item = Row<T, U>;

    fn next(&mut self) -> Option<Row<T, U>> {
        self._state.next_index += 1;
        if self._state.next_index <= self.vals.len() {
            Some(self.expand_row(self._state.next_index - 1))
        } else {
            None
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rows() -> Rows<f32, i32> {
        vec![
            (vec![1, 1, 1, 2], 1.0),
            (vec![1, 1, 1, 3], 2.0),
            (vec![1, 2, 1, 1], 3.0),
            (vec![1, 2, 1, 3], 4.0),
            (vec![1, 2, 2, 1], 5.0),
            (vec![2, 2, 2, 1], 6.0),
            (vec![2, 2, 2, 2], 7.0),
            (vec![2, 2, 2, 3], 8.0),
        ]
    }

    fn sample_csf() -> CompressedSparseFiber<f32, i32> {
        CompressedSparseFiber::new(
            vec![vec![0, 2, 3], vec![0, 1, 3, 4], vec![0, 2, 4, 5, 8]],
            vec![vec![1, 2], vec![1, 2, 2], vec![1, 1, 2, 2], vec![2, 3, 1, 3, 1, 1, 2, 3]],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
    }

    #[test]
    fn test_build() {
        let rows = sample_rows();
        let x = CompressedSparseFiber::from(&rows);
        assert_eq!(x.fids[0], vec![1, 2]);
        assert_eq!(x.fids[1], vec![1, 2, 2]);
        assert_eq!(x.fids[2], vec![1, 1, 2, 2]);
        assert_eq!(x.fids[3], vec![2, 3, 1, 3, 1, 1, 2, 3]);

        assert_eq!(x.fptr[0], vec![0, 2, 3]);
        assert_eq!(x.fptr[1], vec![0, 1, 3, 4]);
        assert_eq!(x.fptr[2], vec![0, 2, 4, 5, 8]);

        assert_eq!(x.vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }


    #[test]
    fn test_expand_row() {
        let x = sample_csf();

        let (key, val) = x.expand_row(0);
        assert_eq!(key, vec![1, 1, 1, 2]);
        assert_eq!(val, 1.0);

        let (key, val) = x.expand_row(4);
        assert_eq!(key, vec![1, 2, 2, 1]);
        assert_eq!(val, 5.0);

        let (key, val) = x.expand_row(6);
        assert_eq!(key, vec![2, 2, 2, 2]);
        assert_eq!(val, 7.0);
    }

    #[test]
    fn test_iterate() {
        let rows = sample_rows();
        let x = CompressedSparseFiber::from(&rows);

        for (vec_out, val_out) in x {
            let (_, value) = sample_rows().into_iter()
                .find(|(vector, _)| vector == &vec_out).unwrap();
            assert_eq!(value, val_out);
        }
    }

    #[test]
    fn test_builder() {
        let mut builder: CompressedSparseFiberBuilder<_, _> = CompressedSparseFiberBuilder::new();
        let rows =  sample_rows();
        for (row, v) in rows {
            builder.rows.insert(&row, v);
        }

        let x = builder.build();

        for (vec_out, val_out) in x {
            let (_, value) = sample_rows().into_iter()
                .find(|(vector, _)| vector == &vec_out).unwrap();
            assert_eq!(value, val_out);
        }
    }

    fn expected_sum(rows: &Rows<f32, i32>, col_index: usize) -> i32 {
        let mut result = 0;
        for (row, _) in rows {
            result += row[col_index];
        }
        result
    }

    #[test]
    fn test_sum() {
        let x = sample_csf();
        let rows = sample_rows();

        assert_eq!(expected_sum(&rows, 0), x.sum_column(0));
        assert_eq!(expected_sum(&rows, 1), x.sum_column(1));
        assert_eq!(expected_sum(&rows, 2), x.sum_column(2));
        assert_eq!(expected_sum(&rows, 3), x.sum_column(3));
    }
}
