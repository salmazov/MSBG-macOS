// Lightweight TBB compatibility layer for environments without oneTBB.
// Provides the minimal API surface used by MSBG, backed by OpenMP/std.
#pragma once

#include <algorithm>
#include <cstddef>
#include <omp.h>

namespace tbb {

// ---------------------------------------------------------------------------
// blocked_range
// ---------------------------------------------------------------------------
template <typename Index>
class blocked_range {
 public:
  blocked_range(Index begin_in, Index end_in) : begin_(begin_in), end_(end_in) {}
  Index begin() const { return begin_; }
  Index end() const { return end_; }

 private:
  Index begin_;
  Index end_;
};

// ---------------------------------------------------------------------------
// parallel_for
// ---------------------------------------------------------------------------
template <typename Index, typename Func>
inline void parallel_for(const blocked_range<Index>& range, const Func& func) {
  Index begin = range.begin();
  Index end = range.end();
#pragma omp parallel for schedule(static)
  for (Index i = begin; i < end; ++i) {
    func(blocked_range<Index>(i, i + 1));
  }
}

// ---------------------------------------------------------------------------
// parallel_sort
// ---------------------------------------------------------------------------
template <typename RandomIt, typename Compare>
inline void parallel_sort(RandomIt first, RandomIt last, Compare comp) {
  std::sort(first, last, comp);
}

// ---------------------------------------------------------------------------
// this_task_arena::current_thread_index
// ---------------------------------------------------------------------------
namespace this_task_arena {
inline int current_thread_index() { return omp_get_thread_num(); }
inline int max_concurrency() { return omp_get_max_threads(); }
}  // namespace this_task_arena

// ---------------------------------------------------------------------------
// global_control shim (only max_allowed_parallelism is used)
// ---------------------------------------------------------------------------
class global_control {
 public:
  enum parameter { max_allowed_parallelism };

  global_control(parameter, int value) { omp_set_num_threads(value); }
};

}  // namespace tbb
