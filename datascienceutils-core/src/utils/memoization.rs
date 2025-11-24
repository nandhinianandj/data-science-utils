//! Memoization utilities for caching function results
//!
//! Provides decorators and utilities for memoizing expensive computations

use cached::proc_macro::cached;
use std::hash::Hash;

/// Memoized version of a computation
///
/// This uses the `cached` crate to automatically cache function results.
/// The cache key is based on the function arguments.
///
/// # Example
/// ```
/// use datascienceutils_core::utils::memoization::memoized_fibonacci;
///
/// let result = memoized_fibonacci(10);
/// assert_eq!(result, 55);
/// ```
#[cached]
pub fn memoized_fibonacci(n: u64) -> u64 {
    if n <= 1 {
        n
    } else {
        memoized_fibonacci(n - 1) + memoized_fibonacci(n - 2)
    }
}

/// Create a memoized version of any function
///
/// This is a generic memoization wrapper that can be used with any function
/// that has hashable arguments and a cloneable return type.
pub struct Memoize<F, Args, Output>
where
    Args: Hash + Eq + Clone,
    Output: Clone,
    F: Fn(Args) -> Output,
{
    func: F,
    cache: std::sync::Mutex<std::collections::HashMap<Args, Output>>,
}

impl<F, Args, Output> Memoize<F, Args, Output>
where
    Args: Hash + Eq + Clone,
    Output: Clone,
    F: Fn(Args) -> Output,
{
    /// Create a new memoized function
    pub fn new(func: F) -> Self {
        Self {
            func,
            cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Call the memoized function
    pub fn call(&self, args: Args) -> Output {
        let mut cache = self.cache.lock().unwrap();
        
        if let Some(result) = cache.get(&args) {
            return result.clone();
        }

        let result = (self.func)(args.clone());
        cache.insert(args, result.clone());
        result
    }

    /// Clear the cache
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        let cache = self.cache.lock().unwrap();
        cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memoized_fibonacci() {
        let result = memoized_fibonacci(10);
        assert_eq!(result, 55);
        
        let result = memoized_fibonacci(20);
        assert_eq!(result, 6765);
    }

    #[test]
    fn test_memoize_wrapper() {
        let expensive_fn = |x: i32| -> i32 {
            // Simulate expensive computation
            x * x
        };

        let memoized = Memoize::new(expensive_fn);
        
        assert_eq!(memoized.call(5), 25);
        assert_eq!(memoized.call(10), 100);
        assert_eq!(memoized.cache_size(), 2);
        
        // Call again with same argument - should use cache
        assert_eq!(memoized.call(5), 25);
        assert_eq!(memoized.cache_size(), 2);
        
        memoized.clear();
        assert_eq!(memoized.cache_size(), 0);
    }
}
