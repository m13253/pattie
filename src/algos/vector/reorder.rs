use std::convert::identity;
use std::mem::{transmute, MaybeUninit};

/// Reorder a vector in place, the order is defined by `order`.
/// After the reordering, the contents in `order` will become `0..len`.
///
/// This algorithm takes `O(n)` time and `n` space.
///
/// ```
/// use pattie::algos::vector::reorder::reorder_forward;
///
/// let array = vec!['H', 'I', 'B', 'F', 'D', 'E', 'C', 'A', 'J', 'G'];
/// let order = vec![7, 2, 6, 4, 5, 3, 9, 0, 1, 8];
/// let result = reorder_forward(&array, &order).into_iter().map(Clone::clone).collect::<Vec<_>>();
/// assert_eq!(result, vec!['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']);
/// ```
pub fn reorder_forward<'a, T>(vec: &'a [T], order: &[usize]) -> Vec<&'a T> {
    let mut result = vec![MaybeUninit::uninit(); order.len()];
    for (r, &o) in result.iter_mut().zip(order.iter()) {
        r.write(&vec[o]);
    }
    unsafe { transmute(result) }
}

/// Reorder a vector in place, the order is defined by `order`.
/// After the reordering, the contents in `order` will become `0..len`.
///
/// This algorithm takes O(n) time.
///
/// ```
/// use pattie::algos::vector::reorder::reorder_forward_inplace;
///
/// let mut array = vec!['H', 'I', 'B', 'F', 'D', 'E', 'C', 'A', 'J', 'G'];
/// let mut order = vec![7, 2, 6, 4, 5, 3, 9, 0, 1, 8];
/// reorder_forward_inplace(&mut array, &mut order);
/// assert_eq!(array, vec!['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']);
/// assert_eq!(order, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
/// ```
pub fn reorder_forward_inplace<T>(vec: &mut [T], order: &mut [usize]) {
    assert_eq!(vec.len(), order.len());
    let len = order.len();
    for i in 0..len {
        let mut u = i;
        let mut v = *unsafe { order.get_unchecked(i) };
        while i != v {
            vec.swap(u, v);
            u = *unsafe { order.get_unchecked(i) };
            assert_ne!(order[i], order[v]);
            order.swap(i, v);
            v = *unsafe { order.get_unchecked(i) };
        }
    }
}

/// Reorder a vector in place, the order is defined by `order`.
/// After the reordering, the contents in `order` will become `0..len`.
///
/// This algorithm takes `O(n)` time and `2*n` space.
///
/// ```
/// use pattie::algos::vector::reorder::reorder_backward;
///
/// let array = vec!['H', 'I', 'B', 'F', 'D', 'E', 'C', 'A', 'J', 'G'];
/// let order = vec![7, 8, 1, 5, 3, 4, 2, 0, 9, 6];
/// let result = reorder_backward(&array, &order).into_iter().map(Clone::clone).collect::<Vec<_>>();
/// assert_eq!(result, vec!['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']);
/// ```
pub fn reorder_backward<'a, T>(vec: &'a [T], order: &[usize]) -> Vec<&'a T> {
    let mut result = vec![MaybeUninit::uninit(); vec.len()];
    let mut init = vec![false; vec.len()];
    for (v, &o) in vec.iter().zip(order.iter()) {
        if init[o] {
            panic!("duplicate index");
        }
        result[o].write(v);
        init[o] = true;
    }
    if !init.into_iter().all(identity) {
        panic!("missing index");
    }
    unsafe { transmute(result) }
}

/// Reorder a vector in place, the order is defined by `order`.
/// After the reordering, the contents in `order` will become `0..len`.
///
/// This algorithm takes O(n) time.
///
/// ```
/// use pattie::algos::vector::reorder::reorder_backward_inplace;
///
/// let mut array = vec!['H', 'I', 'B', 'F', 'D', 'E', 'C', 'A', 'J', 'G'];
/// let mut order = vec![7, 8, 1, 5, 3, 4, 2, 0, 9, 6];
/// reorder_backward_inplace(&mut array, &mut order);
/// assert_eq!(array, vec!['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']);
/// assert_eq!(order, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
/// ```
pub fn reorder_backward_inplace<T>(vec: &mut [T], order: &mut [usize]) {
    assert_eq!(vec.len(), order.len());
    let len = order.len();
    for i in 0..len {
        let mut v = *unsafe { order.get_unchecked(i) };
        while i != v {
            vec.swap(i, v);
            assert_ne!(order[i], order[v]);
            order.swap(i, v);
            v = *unsafe { order.get_unchecked(i) };
        }
    }
}
