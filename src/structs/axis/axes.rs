use super::Axis;
use smallvec::SmallVec;

const SMALL_DIMS: usize = 4;

pub type Axes<IT> = SmallVec<[Axis<IT>; SMALL_DIMS]>;
