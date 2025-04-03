use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[pyfunction]
fn calculate_average(numbers_str: &str) -> PyResult<f64> {
    // Parse the comma-separated string into numbers
    let numbers: Result<Vec<f64>, _> = numbers_str
        .split(',')
        .map(|s| s.trim().parse::<f64>())
        .collect();

    // Handle parsing errors
    let numbers = match numbers {
        Ok(nums) => nums,
        Err(_) => return Err(PyValueError::new_err("Could not parse numbers"))
    };

    // Check if the vector is empty
    if numbers.is_empty() {
        return Err(PyValueError::new_err("Empty list of numbers"));
    }

    // Calculate the average
    let sum: f64 = numbers.iter().sum();
    let avg = sum / numbers.len() as f64;

    Ok(avg)
}

#[pymodule]
fn rust_math_extensions(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_average, m)?)?;
    Ok(())
}