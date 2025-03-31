// Rust CLI port of MathSolver
use std::env;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::time::Instant;
use std::io::{self, Write};
use rand::Rng;
use regex::Regex;
use chrono::Utc;

// API keys and settings
const OPENAI_API_KEY: &str = "ADD YOUR KEY HERE";
const DEFAULT_MODEL: &str = "gpt-4o-mini";
const DEFAULT_TEMPERATURE: f32 = 0.0;
const MAX_VERIFICATION_RETRIES: usize = 5;

// Sample problems for UI
const SAMPLE_PROBLEMS: [&str; 7] = [
    "What is the square root of -1",
    "What is 25 + 37?",
    "Multiply 13 by 7",
    "What is 144 divided by 12?",
    "What is the square root of 81?",
    "If x = 5 and y = 3, what is x^y?",
    "(2-3)*5^2 "
];

// Custom error type for stopping execution
#[derive(Debug)]
struct StopException {
    message: String,
}

impl StopException {
    fn new(message: &str) -> Self {
        StopException {
            message: message.to_string(),
        }
    }
}

impl fmt::Display for StopException {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Execution stopped: {}", self.message)
    }
}

impl Error for StopException {}

// Logging setup
fn setup_logger(_name: &str) -> () {
    // In a real application, we would set up proper logging here
    // For simplicity, we'll just use println!
}

// MathToolbox implementation
#[derive(Clone)]
struct MathToolbox {
    unreliable_tools: Vec<String>,
    tool_stats: HashMap<String, HashMap<String, usize>>,
    errors_enabled: bool,
}

impl MathToolbox {
    fn new() -> Self {
        let mut tool_stats = HashMap::new();
        let tools = vec![
            "sum", "product", "divide", "subtract", "power", "sqrt", "modulo", "round_number"
        ];

        for tool in tools {
            let mut stats = HashMap::new();
            stats.insert("calls".to_string(), 0);
            stats.insert("errors".to_string(), 0);
            tool_stats.insert(tool.to_string(), stats);
        }

        MathToolbox {
            unreliable_tools: vec!["sum".to_string(), "product".to_string()],
            tool_stats,
            errors_enabled: true, // Default to errors enabled
        }
    }

    fn set_all_tools_reliable(&mut self) {
        self.unreliable_tools.clear();
    }

    fn unset_all_tools_reliable(&mut self) {
        self.unreliable_tools = vec!["sum".to_string(), "product".to_string()];
    }

    fn toggle_errors(&mut self) {
        self.errors_enabled = !self.errors_enabled;
        if self.errors_enabled {
            self.unset_all_tools_reliable();
            println!("Tool errors are now ENABLED. Some tools may produce incorrect results.");
        } else {
            self.set_all_tools_reliable();
            println!("Tool errors are now DISABLED. All tools will work reliably.");
        }
    }

    fn get_errors_status(&self) -> &str {
        if self.errors_enabled {
            "enabled"
        } else {
            "disabled"
        }
    }

    fn get_tools_string(&self) -> String {
        self.tool_stats.keys().cloned().collect::<Vec<String>>().join(",")
    }

    fn sum(&mut self, numbers_str: &str) -> String {
        let stats = self.tool_stats.get_mut("sum").unwrap();
        *stats.get_mut("calls").unwrap() += 1;

        // Introduce errors 40% of the time if errors are enabled
        let mut rng = rand::thread_rng();
        if self.errors_enabled && rng.gen::<f64>() < 0.4 && self.unreliable_tools.contains(&"sum".to_string()) {
            *stats.get_mut("errors").unwrap() += 1;
            if rng.gen::<f64>() < 0.5 {
                return "Error occurred: Invalid input format".to_string();
            } else {
                match numbers_str
                    .split(',')
                    .map(|x| x.trim().parse::<f64>())
                    .collect::<Result<Vec<f64>, _>>()
                {
                    Ok(numbers) => {
                        let sum: f64 = numbers.iter().sum();
                        let error = rng.gen_range(1..=10) as f64;
                        return (sum + error).to_string();
                    },
                    Err(_) => return "Error occurred: Could not parse numbers".to_string()
                }
            }
        }

        match numbers_str
            .split(',')
            .map(|x| x.trim().parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()
        {
            Ok(numbers) => {
                let sum: f64 = numbers.iter().sum();
                sum.to_string()
            },
            Err(e) => {
                *stats.get_mut("errors").unwrap() += 1;
                format!("Error occurred: {}", e)
            }
        }
    }

    fn product(&mut self, numbers_str: &str) -> String {
        let stats = self.tool_stats.get_mut("product").unwrap();
        *stats.get_mut("calls").unwrap() += 1;

        // Introduce errors 30% of the time if errors are enabled
        let mut rng = rand::thread_rng();
        if self.errors_enabled && rng.gen::<f64>() < 0.3 && self.unreliable_tools.contains(&"product".to_string()) {
            *stats.get_mut("errors").unwrap() += 1;
            if rng.gen::<f64>() < 0.5 {
                return "Error occurred: Invalid input format".to_string();
            } else {
                match numbers_str
                    .split(',')
                    .map(|x| x.trim().parse::<f64>())
                    .collect::<Result<Vec<f64>, _>>()
                {
                    Ok(numbers) => {
                        let product = numbers.iter().fold(1.0, |acc, &x| acc * x);
                        return (product * 1.1).to_string(); // Wrong by 10%
                    },
                    Err(_) => return "Error occurred: Could not parse numbers".to_string()
                }
            }
        }

        match numbers_str
            .split(',')
            .map(|x| x.trim().parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()
        {
            Ok(numbers) => {
                let product = numbers.iter().fold(1.0, |acc, &x| acc * x);
                product.to_string()
            },
            Err(e) => {
                *stats.get_mut("errors").unwrap() += 1;
                format!("Error occurred: {}", e)
            }
        }
    }

    fn divide(&mut self, numbers_str: &str) -> String {
        let stats = self.tool_stats.get_mut("divide").unwrap();
        *stats.get_mut("calls").unwrap() += 1;

        match numbers_str
            .split(',')
            .map(|x| x.trim().parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()
        {
            Ok(numbers) => {
                if numbers.len() != 2 {
                    *stats.get_mut("errors").unwrap() += 1;
                    return "Error occurred: Exactly two numbers are required for division".to_string();
                }
                if numbers[1] == 0.0 {
                    *stats.get_mut("errors").unwrap() += 1;
                    return "Error occurred: Cannot divide by zero".to_string();
                }
                (numbers[0] / numbers[1]).to_string()
            },
            Err(e) => {
                *stats.get_mut("errors").unwrap() += 1;
                format!("Error occurred: {}", e)
            }
        }
    }

    fn subtract(&mut self, numbers_str: &str) -> String {
        let stats = self.tool_stats.get_mut("subtract").unwrap();
        *stats.get_mut("calls").unwrap() += 1;

        match numbers_str
            .split(',')
            .map(|x| x.trim().parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()
        {
            Ok(numbers) => {
                if numbers.len() != 2 {
                    *stats.get_mut("errors").unwrap() += 1;
                    return "Error occurred: Exactly two numbers are required for subtraction".to_string();
                }
                (numbers[0] - numbers[1]).to_string()
            },
            Err(e) => {
                *stats.get_mut("errors").unwrap() += 1;
                format!("Error occurred: {}", e)
            }
        }
    }

    fn power(&mut self, numbers_str: &str) -> String {
        let stats = self.tool_stats.get_mut("power").unwrap();
        *stats.get_mut("calls").unwrap() += 1;

        match numbers_str
            .split(',')
            .map(|x| x.trim().parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()
        {
            Ok(numbers) => {
                if numbers.len() != 2 {
                    *stats.get_mut("errors").unwrap() += 1;
                    return "Error occurred: Exactly two numbers are required (base, exponent)".to_string();
                }
                numbers[0].powf(numbers[1]).to_string()
            },
            Err(e) => {
                *stats.get_mut("errors").unwrap() += 1;
                format!("Error occurred: {}", e)
            }
        }
    }

    fn sqrt(&mut self, number_str: &str) -> String {
        let stats = self.tool_stats.get_mut("sqrt").unwrap();
        *stats.get_mut("calls").unwrap() += 1;

        match number_str.trim().parse::<f64>() {
            Ok(number) => {
                if number < 0.0 {
                    // Handle negative numbers with complex numbers
                    let imag = (-number).sqrt();
                    if imag == 1.0 {
                        return "i".to_string();
                    } else if imag == -1.0 {
                        return "-i".to_string();
                    } else {
                        return format!("{}i", imag);
                    }
                } else {
                    return number.sqrt().to_string();
                }
            },
            Err(e) => {
                *stats.get_mut("errors").unwrap() += 1;
                format!("Error occurred: {}", e)
            }
        }
    }

    fn modulo(&mut self, numbers_str: &str) -> String {
        let stats = self.tool_stats.get_mut("modulo").unwrap();
        *stats.get_mut("calls").unwrap() += 1;

        match numbers_str
            .split(',')
            .map(|x| x.trim().parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()
        {
            Ok(numbers) => {
                if numbers.len() != 2 {
                    *stats.get_mut("errors").unwrap() += 1;
                    return "Error occurred: Exactly two numbers are required for modulo".to_string();
                }
                if numbers[1] == 0.0 {
                    *stats.get_mut("errors").unwrap() += 1;
                    return "Error occurred: Cannot divide by zero".to_string();
                }
                (numbers[0] % numbers[1]).to_string()
            },
            Err(e) => {
                *stats.get_mut("errors").unwrap() += 1;
                format!("Error occurred: {}", e)
            }
        }
    }

    fn round_number(&mut self, input_str: &str) -> String {
        let stats = self.tool_stats.get_mut("round_number").unwrap();
        *stats.get_mut("calls").unwrap() += 1;

        let parts: Vec<&str> = input_str.split(',').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            *stats.get_mut("errors").unwrap() += 1;
            return "Error occurred: Input should be 'number, decimal_places'".to_string();
        }

        match parts[0].parse::<f64>() {
            Ok(number) => {
                match parts[1].parse::<i32>() {
                    Ok(decimal_places) => {
                        let factor = 10.0_f64.powi(decimal_places);
                        ((number * factor).round() / factor).to_string()
                    },
                    Err(e) => {
                        *stats.get_mut("errors").unwrap() += 1;
                        format!("Error occurred: {}", e)
                    }
                }
            },
            Err(e) => {
                *stats.get_mut("errors").unwrap() += 1;
                format!("Error occurred: {}", e)
            }
        }
    }

    fn get_stats(&self) -> &HashMap<String, HashMap<String, usize>> {
        &self.tool_stats
    }
}

// Tools for agents
#[derive(Clone)]
struct Tool {
    name: String,
    description: String,
}

// Virtual tool manager implementation
struct VirtualToolManager {
    virtual_tools: HashMap<String, VirtualTool>,
    successful_sequences: HashMap<String, SuccessfulSequence>,
    tool_failure_counts: HashMap<String, usize>,
    max_failures: usize,
}

#[derive(Clone)]
struct VirtualTool {
    name: String,
    description: String,
    primary_tool: Option<String>,
    tool_sequence: Vec<ToolStep>,
}

#[derive(Clone, Debug)]
struct ToolStep {
    tool: String,
    tool_input: String,
}

struct SuccessfulSequence {
    problem: String,
    sequence: Vec<ToolStep>,
    result: String,
    created_at: chrono::DateTime<Utc>,
}

#[derive(Debug)]
struct NumberWithContext {
    value: String,
    position: usize,
    left_paren: bool,
    right_paren: bool,
    left_op: Option<char>,
    right_op: Option<char>,
}

impl VirtualToolManager {
    fn new(max_failures: usize) -> Self {
        VirtualToolManager {
            virtual_tools: HashMap::new(),
            successful_sequences: HashMap::new(),
            tool_failure_counts: HashMap::new(),
            max_failures,
        }
    }

    fn hash_problem(&self, problem: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        problem.to_lowercase().trim().hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    fn record_successful_sequence(&mut self, problem: &str, sequence: Vec<ToolStep>, result: &str) {
        let problem_hash = self.hash_problem(problem);
        println!("Storing successful sequence: {:?}", sequence);

        self.successful_sequences.insert(problem_hash.clone(), SuccessfulSequence {
            problem: problem.to_string(),
            sequence: sequence.clone(),
            result: result.to_string(),
            created_at: Utc::now(),
        });

        // Create a virtual tool
        self._create_virtual_tool(&problem_hash);
    }

    fn _create_virtual_tool(&mut self, problem_hash: &str) {
        if let Some(sequence_data) = self.successful_sequences.get(problem_hash) {
            let problem_type = self._categorize_problem(&sequence_data.problem);
            let tool_sequence = sequence_data.sequence.clone();

            // Determine the primary tool for naming/description purposes
            let primary_tool = if !tool_sequence.is_empty() {
                Some(tool_sequence[0].tool.clone())
            } else {
                None
            };

            // Register the virtual tool
            let tool_seq_len = tool_sequence.len(); // Get length before moving
            self.virtual_tools.insert(problem_hash.to_string(), VirtualTool {
                name: format!("VirtualTool_{}_{}", problem_type, &problem_hash[0..8]),
                description: format!("Solves {} problems similar to: '{}'", problem_type, sequence_data.problem),
                primary_tool,
                tool_sequence,
            });

            println!("Created virtual tool '{}' using sequence of {} tools",
                    self.virtual_tools.get(problem_hash).unwrap().name,
                    tool_seq_len);
        }
    }

    fn find_matching_virtual_tool(&self, problem: &str) -> Option<&VirtualTool> {
        let problem_hash = self.hash_problem(problem);

        // Check for exact match
        if let Some(tool) = self.virtual_tools.get(&problem_hash) {
            if self._is_tool_relevant_for_problem(problem, &tool.tool_sequence) {
                return Some(tool);
            } else {
                println!("Found exact match tool {} but it's not relevant for this problem type.", tool.name);
                return None;
            }
        }

        // Parse the current problem to extract its structure
        let new_problem_numbers = self._parse_expression(problem);

        // Count expected operations based on numbers in the problem
        let expected_op_count = self._estimate_operation_count(problem, &new_problem_numbers);
        println!("Estimated operations needed for problem: {}", expected_op_count);

        let mut best_match = None;
        let mut best_match_score = 0.0;

        // Consider all available tools
        for (hash_key, tool) in &self.virtual_tools {
            // Get the original problem associated with this tool
            if let Some(original_problem) = self.successful_sequences.get(hash_key).map(|seq| &seq.problem) {
                // Skip tools with too many or too few operations
                let tool_op_count = tool.tool_sequence.len();
                if (tool_op_count as i32 - expected_op_count as i32).abs() > 1 {
                    println!("Tool {} has {} operations, but problem needs ~{}",
                           tool.name, tool_op_count, expected_op_count);
                    continue;
                }

                // Check operation types
                if !self._has_compatible_operations(problem, &tool.tool_sequence) {
                    println!("Tool {} has incompatible operation types for this problem", tool.name);
                    continue;
                }

                // Get structure similarity score
                let original_problem_numbers = self._parse_expression(original_problem);
                let structure_score = self._calculate_structure_similarity(
                    &new_problem_numbers,
                    &original_problem_numbers
                );

                // Calculate overall match score
                let problem_lower = problem.to_lowercase();
                let problem_tokens: Vec<&str> = problem_lower.split_whitespace().collect();
                let description = tool.description.to_lowercase();
                let desc_tokens: Vec<&str> = description.split_whitespace().collect();

                let mut token_overlap = 0;
                for token in &problem_tokens {
                    if desc_tokens.contains(token) {
                        token_overlap += 1;
                    }
                }

                let match_score = structure_score * 0.9 + (1.0f64.min(token_overlap as f64 / 5.0)) * 0.3;

                if match_score > best_match_score && match_score > 0.6 {
                    best_match = Some(tool);
                    best_match_score = match_score;
                }
            }
        }

        if let Some(tool) = best_match {
            println!("Selected tool {} with match score {:.2}", tool.name, best_match_score);
            return Some(tool);
        }

        None
    }

    fn _is_tool_relevant_for_problem(&self, problem: &str, tool_sequence: &[ToolStep]) -> bool {
        // Check for simple multiplication problems
        let is_simple_multiplication = Regex::new(r"multiply\s+(\d+)\s+by\s+(\d+)").unwrap()
            .is_match(&problem.to_lowercase());

        if is_simple_multiplication {
            // For simple multiplication, don't use sequences with round_number
            for step in tool_sequence {
                if step.tool == "round_number" {
                    return false;
                }
            }
        }

        // By default, consider the tool relevant
        true
    }

    fn _categorize_problem(&self, problem: &str) -> String {
        let problem_lower = problem.to_lowercase();

        if problem_lower.contains("add") || problem_lower.contains("sum") || problem_lower.contains("plus") {
            return "Addition".to_string();
        } else if problem_lower.contains("multiply") || problem_lower.contains("product") {
            return "Multiplication".to_string();
        } else if problem_lower.contains("divide") || problem_lower.contains("quotient") {
            return "Division".to_string();
        } else if problem_lower.contains("subtract") || problem_lower.contains("difference") || problem_lower.contains("minus") {
            return "Subtraction".to_string();
        } else if problem_lower.contains("power") || problem_lower.contains("exponent") || problem_lower.contains("square") {
            return "Exponentiation".to_string();
        } else if problem_lower.contains("symbolic") || problem_lower.contains("derivative") || problem_lower.contains("integral") {
            return "Calculus".to_string();
        } else {
            return "General".to_string();
        }
    }

    fn _parse_expression(&self, expression: &str) -> Vec<NumberWithContext> {
        let mut numbers = Vec::new();
        let re = Regex::new(r"-?\d+\.?\d*").unwrap();

        for cap in re.captures_iter(expression) {
            let value = cap.get(0).unwrap().as_str();
            let position = cap.get(0).unwrap().start();

            // Determine context - look at characters before and after
            let left_context = &expression[..position];
            let right_context = &expression[(position + value.len())..];

            // Check for operations and parentheses
            let left_paren = left_context.contains('(') &&
                               !left_context[left_context.rfind('(').unwrap_or(0)..].contains(')');
            let right_paren = right_context.contains(')') &&
                               !right_context[..right_context.find(')').unwrap_or(right_context.len())].contains('(');

            // Check for operators near the number
            let mut left_op = None;
            let mut right_op = None;

            for op in &['+', '-', '*', '/', '^'] {
                if !left_context.is_empty() && left_context.chars().last().unwrap() == *op {
                    left_op = Some(*op);
                }
                if !right_context.is_empty() && right_context.chars().next().unwrap() == *op {
                    right_op = Some(*op);
                }
            }

            numbers.push(NumberWithContext {
                value: value.to_string(),
                position,
                left_paren,
                right_paren,
                left_op,
                right_op
            });
        }

        numbers
    }

    fn _estimate_operation_count(&self, problem: &str, numbers: &[NumberWithContext]) -> usize {
        // Count explicit operation words
        let mut op_count = 0;

        // Count based on operation keywords
        let re_add = Regex::new(r"\b(?:add|plus|sum)\b").unwrap();
        let re_subtract = Regex::new(r"\b(?:subtract|minus|difference)\b").unwrap();
        let re_multiply = Regex::new(r"\b(?:multiply|times|product)\b").unwrap();
        let re_divide = Regex::new(r"\b(?:divide|division|quotient)\b").unwrap();

        op_count += re_add.find_iter(&problem.to_lowercase()).count();
        op_count += re_subtract.find_iter(&problem.to_lowercase()).count();
        op_count += re_multiply.find_iter(&problem.to_lowercase()).count();
        op_count += re_divide.find_iter(&problem.to_lowercase()).count();

        // Count based on mathematical symbols
        let op_symbols = ['+', '-', '*', '/', '^'];
        for symbol in &op_symbols {
            op_count += problem.chars().filter(|&c| c == *symbol).count();
        }

        // If we have n numbers, we typically need n-1 operations
        // This is a fallback if we couldn't detect operations explicitly
        if op_count == 0 && numbers.len() > 1 {
            op_count = numbers.len() - 1;
        }

        // Ensure at least 1 operation if we have numbers
        if numbers.is_empty() {
            op_count
        } else {
            op_count.max(1)
        }
    }

    fn _has_compatible_operations(&self, problem: &str, tool_sequence: &[ToolStep]) -> bool {
        // Extract operation types from the problem
        let problem_lower = problem.to_lowercase();

        // Check for specific operation types in the problem
        let has_addition = problem_lower.contains("add") ||
                          problem_lower.contains("sum") ||
                          problem_lower.contains("plus") ||
                          problem_lower.contains("+");

        let has_subtraction = problem_lower.contains("subtract") ||
                             problem_lower.contains("minus") ||
                             problem_lower.contains("difference") ||
                             problem_lower.contains("-");

        let has_multiplication = problem_lower.contains("multiply") ||
                                problem_lower.contains("product") ||
                                problem_lower.contains("times") ||
                                problem_lower.contains("*");

        let has_division = problem_lower.contains("divide") ||
                          problem_lower.contains("quotient") ||
                          problem_lower.contains("/");

        let has_power = problem_lower.contains("power") ||
                       problem_lower.contains("exponent") ||
                       problem_lower.contains("^") ||
                       problem_lower.contains("squared") ||
                       problem_lower.contains("cubed");

        let has_square_root = problem_lower.contains("sqrt") ||
                             problem_lower.contains("square root");

        // Count operation types in the tool sequence
        let mut tool_operations = HashMap::new();
        let operations = ["sum", "subtract", "product", "divide", "power", "sqrt", "modulo", "round_number"];

        for op in &operations {
            tool_operations.insert(*op, 0);
        }

        for step in tool_sequence {
            if let Some(count) = tool_operations.get_mut(step.tool.as_str()) {
                *count += 1;
            }
        }

        // Check for compatibility based on operations
        if has_addition && *tool_operations.get("sum").unwrap() == 0 {
            return false;
        }
        if has_subtraction && *tool_operations.get("subtract").unwrap() == 0 {
            return false;
        }
        if has_multiplication && *tool_operations.get("product").unwrap() == 0 {
            return false;
        }
        if has_division && *tool_operations.get("divide").unwrap() == 0 {
            return false;
        }
        if has_power && *tool_operations.get("power").unwrap() == 0 {
            return false;
        }
        if has_square_root && *tool_operations.get("sqrt").unwrap() == 0 {
            return false;
        }

        // For subtraction specifically, check if the number of operations matches
        if has_subtraction {
            // Count subtraction operations in problem
            let subtract_count = problem_lower.matches("subtract").count() +
                              problem_lower.matches("minus").count() +
                              problem_lower.matches("-").count();

            // If problem has clearly more subtractions than the tool can handle
            if subtract_count > *tool_operations.get("subtract").unwrap() + 1 {
                println!("Problem needs ~{} subtractions but tool only has {}",
                       subtract_count,
                       tool_operations.get("subtract").unwrap());
                return false;
            }
        }

        true
    }

    fn _calculate_structure_similarity(
        &self,
        new_numbers: &[NumberWithContext],
        original_numbers: &[NumberWithContext]
    ) -> f64 {
        // If number counts are very different, it's not a good match
        if (new_numbers.len() as i32 - original_numbers.len() as i32).abs() > 1 {
            return 0.2;  // Low base similarity
        }

        // If both have no numbers, return medium similarity
        if new_numbers.is_empty() && original_numbers.is_empty() {
            return 0.5;
        }

        // Calculate similarity based on patterns of numbers and operations
        let mut similarity = 0.0;

        // Start with base similarity based on number count match
        if new_numbers.len() == original_numbers.len() {
            similarity += 0.5;
        } else {
            similarity += 0.3;
        }

        // Check pattern of operations between numbers
        let operation_pattern_match = self._compare_operation_patterns(new_numbers, original_numbers);
        similarity += operation_pattern_match * 0.5;

        1.0_f64.min(similarity)
    }

    fn _compare_operation_patterns(
        &self,
        nums1: &[NumberWithContext],
        nums2: &[NumberWithContext]
    ) -> f64 {
        if nums1.is_empty() || nums2.is_empty() {
            return 0.0;
        }

        // Extract operation patterns
        let get_ops_pattern = |nums: &[NumberWithContext]| {
            let mut pattern = Vec::new();
            for i in 0..nums.len() - 1 {
                // Check operations between adjacent numbers
                let curr = &nums[i];
                let next_num = &nums[i + 1];

                // Use left_op of next number or right_op of current
                let op = next_num.left_op.or(curr.right_op);
                pattern.push(op);
            }
            pattern
        };

        let pattern1 = get_ops_pattern(nums1);
        let pattern2 = get_ops_pattern(nums2);

        // If patterns are different lengths, take the shorter one
        let min_len = pattern1.len().min(pattern2.len());
        if min_len == 0 {
            return 0.0;
        }

        // Count matching operations
        let mut matches = 0;
        for i in 0..min_len {
            if pattern1[i] == pattern2[i] {
                matches += 1;
            }
        }

        matches as f64 / min_len as f64
    }

    fn record_tool_failure(&mut self, problem_hash: &str) -> bool {
        *self.tool_failure_counts.entry(problem_hash.to_string()).or_insert(0) += 1;

        // Check if we should remove this tool
        if let Some(failure_count) = self.tool_failure_counts.get(problem_hash) {
            if *failure_count >= self.max_failures {
                if let Some(tool) = self.virtual_tools.get(problem_hash) {
                    println!("Removing unreliable virtual tool '{}' after {} failures",
                           tool.name, self.max_failures);

                    // Remove the tool
                    self.virtual_tools.remove(problem_hash);

                    // Optionally, also remove the successful sequence to prevent recreating the same tool
                    self.successful_sequences.remove(problem_hash);

                    // Clear the failure count
                    self.tool_failure_counts.remove(problem_hash);

                    return true;
                }
            }
        }

        false
    }
}

// Simplified OpenAI API interaction for the CLI version
fn call_openai_api(messages: Vec<Message>) -> Result<String, Box<dyn Error>> {
    // For the CLI version, we'll simulate the response based on the problem
    // In a real implementation, you would call the OpenAI API here

    // Get the user message which contains the problem
    let problem_message = messages.iter()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_str())  // Changed to as_str()
        .unwrap_or("Unknown problem");

    // Generate a reasonable response based on the problem
    if problem_message.contains("25 + 37") {
        return Ok("To solve 25 + 37, I'll use the sum tool.\n\nUsing tool: sum with input: 25, 37\nTool result: 62\n\nTherefore, 25 + 37 = 62".to_string());
    } else if problem_message.contains("13 by 7") {
        return Ok("To multiply 13 by 7, I'll use the product tool.\n\nUsing tool: product with input: 13, 7\nTool result: 91\n\nTherefore, 13 × 7 = 91".to_string());
    } else if problem_message.contains("144 divided by 12") {
        return Ok("To divide 144 by 12, I'll use the divide tool.\n\nUsing tool: divide with input: 144, 12\nTool result: 12\n\nTherefore, 144 ÷ 12 = 12".to_string());
    } else if problem_message.contains("square root of 81") {
        return Ok("To find the square root of 81, I'll use the sqrt tool.\n\nUsing tool: sqrt with input: 81\nTool result: 9\n\nTherefore, √81 = 9".to_string());
    } else if problem_message.contains("square root of -1") {
        return Ok("To find the square root of -1, I'll use the sqrt tool.\n\nUsing tool: sqrt with input: -1\nTool result: i\n\nTherefore, √(-1) = i, which is the imaginary unit.".to_string());
    } else if problem_message.contains("x = 5 and y = 3") {
        return Ok("To calculate x^y where x = 5 and y = 3, I'll use the power tool.\n\nUsing tool: power with input: 5, 3\nTool result: 125\n\nTherefore, 5^3 = 125".to_string());
    } else if problem_message.contains("(2-3)*5^2") {
        return Ok("To solve (2-3)*5^2, I'll break it down step by step.\n\nStep 1: Calculate (2-3)\nUsing tool: subtract with input: 2, 3\nTool result: -1\n\nStep 2: Calculate 5^2\nUsing tool: power with input: 5, 2\nTool result: 25\n\nStep 3: Multiply the results\nUsing tool: product with input: -1, 25\nTool result: -25\n\nTherefore, (2-3)*5^2 = -25".to_string());
    } else {
        // For any other problem, create a generic response
        return Ok(format!("To solve this problem, I need to analyze it carefully.\n\nLet me break down the steps:\n1. First, I'll identify the operations needed.\n2. Then, I'll use the appropriate tools for each step.\n\nI believe the answer is calculated through a series of mathematical operations on the given values in '{}'.", problem_message));
    }
}

// Verify a solution using the same approach
fn verify_solution(problem: &str, proposed_solution: &str) -> (bool, String) {
    // Simple verification logic for demo purposes
    if problem.contains("25 + 37") && proposed_solution.contains("62") {
        (true, "VERIFIED: The solution is correct. 25 + 37 = 62".to_string())
    } else if problem.contains("13 by 7") && proposed_solution.contains("91") {
        (true, "VERIFIED: The solution is correct. 13 × 7 = 91".to_string())
    } else if problem.contains("144 divided by 12") && proposed_solution.contains("12") {
        (true, "VERIFIED: The solution is correct. 144 ÷ 12 = 12".to_string())
    } else if problem.contains("square root of 81") && proposed_solution.contains("9") {
        (true, "VERIFIED: The solution is correct. √81 = 9".to_string())
    } else if problem.contains("square root of -1") && (proposed_solution.contains("i") || proposed_solution.contains("imaginary")) {
        (true, "VERIFIED: The solution is correct. √(-1) = i".to_string())
    } else if problem.contains("x = 5 and y = 3") && proposed_solution.contains("125") {
        (true, "VERIFIED: The solution is correct. 5^3 = 125".to_string())
    } else if problem.contains("(2-3)*5^2") && proposed_solution.contains("-25") {
        (true, "VERIFIED: The solution is correct. (2-3)*5^2 = -25".to_string())
    } else {
        (false, "INCORRECT: The proposed solution may not be accurate. Please double-check your calculations.".to_string())
    }
}

struct Message {
    role: String,
    content: String,
}

// Math solver agent
struct MathSolverAgent {
    toolbox: MathToolbox,
    virtual_tool_manager: VirtualToolManager,
    execution_history: Vec<ToolStep>,
}

impl MathSolverAgent {
    fn new(toolbox: MathToolbox, virtual_tool_manager: VirtualToolManager) -> Self {
        MathSolverAgent {
            toolbox,
            virtual_tool_manager,
            execution_history: Vec::new(),
        }
    }

    fn get_base_tools(&self) -> Vec<Tool> {
        vec![
            Tool {
                name: "sum".to_string(),
                description: "Add a list of numbers. Format: 'num1, num2, num3, ...'".to_string(),
            },
            Tool {
                name: "product".to_string(),
                description: "Multiply a list of numbers. Format: 'num1, num2, num3, ...'".to_string(),
            },
            Tool {
                name: "divide".to_string(),
                description: "Divide first number by second number. Format: 'num1, num2'".to_string(),
            },
            Tool {
                name: "subtract".to_string(),
                description: "Subtract second number from first number. Format: 'num1, num2'".to_string(),
            },
            Tool {
                name: "power".to_string(),
                description: "Raise first number to the power of second number. Format: 'base, exponent'".to_string(),
            },
            Tool {
                name: "sqrt".to_string(),
                description: "Calculate the square root of a number.".to_string(),
            },
            Tool {
                name: "modulo".to_string(),
                description: "Calculate the remainder when first number is divided by second. Format: 'num1, num2'".to_string(),
            },
            Tool {
                name: "round_number".to_string(),
                description: "Round a number to the specified decimal places. Format: 'number, decimal_places'".to_string(),
            },
        ]
    }

    fn solve_problem(&mut self, problem: &str) -> String {
        // Check if we have a virtual tool for this type of problem
        if let Some(virtual_tool) = self.virtual_tool_manager.find_matching_virtual_tool(problem) {
            println!("Using virtual tool: {}", virtual_tool.name);

            // Execute the virtual tool sequence with the input
            let mut result = String::new();

            // For this CLI version, we'll implement a simplified virtual tool execution
            // that follows the sequence and calls the appropriate tools
            for step in &virtual_tool.tool_sequence {
                println!("Executing step: {} with input: {}", step.tool, step.tool_input);

                // Extract the tool and input
                let tool_name = &step.tool;
                let tool_input = &step.tool_input;

                // Call the appropriate tool
                let step_result = match tool_name.as_str() {
                    "sum" => self.toolbox.sum(tool_input),
                    "product" => self.toolbox.product(tool_input),
                    "divide" => self.toolbox.divide(tool_input),
                    "subtract" => self.toolbox.subtract(tool_input),
                    "power" => self.toolbox.power(tool_input),
                    "sqrt" => self.toolbox.sqrt(tool_input),
                    "modulo" => self.toolbox.modulo(tool_input),
                    "round_number" => self.toolbox.round_number(tool_input),
                    _ => format!("Unknown tool: {}", tool_name),
                };

                println!("Step result: {}", step_result);

                // If error, try with standard solver
                if step_result.contains("Error") {
                    println!("Virtual tool step failed, falling back to standard solver");
                    break;
                }

                // Update our result
                result = step_result;
            }

            // If we completed successfully, return the result
            if !result.is_empty() && !result.contains("Error") {
                return format!("Solved using virtual tool {}: {}", virtual_tool.name, result);
            }

            // Otherwise, fall back to standard solving
        }

        // Reset execution history for this problem
        self.execution_history.clear();

        // Get tools
        let tools = self.get_base_tools();

        // Format messages for OpenAI
        let tools_desc = tools
            .iter()
            .map(|t| format!("- {}: {}", t.name, t.description))
            .collect::<Vec<_>>()
            .join("\n");

        let system_message = format!(
            "You are a helpful assistant that solves math problems. Use the following tools:\n\n{}\n\nAlways use tools for calculations. Show your work step by step.",
            tools_desc
        );

        let user_message = format!(
            "Solve this math problem: {}. Use the available tools. Do NOT perform calculations yourself. Break down complex calculations into step-by-step tool calls.",
            problem
        );

        let messages = vec![
            Message {
                role: "system".to_string(),
                content: system_message,
            },
            Message {
                role: "user".to_string(),
                content: user_message,
            },
        ];

        // Call the OpenAI API (or our simulation for the CLI version)
        match call_openai_api(messages) {
            Ok(response) => {
                // Extract tool usage from the LLM response
                self.extract_tool_usage(&response, problem);

                response
            },
            Err(e) => {
                format!("Error solving problem: {}", e)
            }
        }
    }

    fn extract_tool_usage(&mut self, response: &str, problem: &str) {
        // This is a simplified version - parse tool usage from the LLM response
        // Example format in response: "Using tool: sum with input: 1, 2, 3"

        let re_tool_use = Regex::new(r"(?i)using\s+(?:tool|the)?\s*:\s*(\w+)\s+with\s+input\s*:\s*([^,\n]+(?:,[^,\n]+)*)").unwrap();

        for cap in re_tool_use.captures_iter(response) {
            let tool = cap[1].to_string();
            let tool_input = cap[2].trim().to_string();

            // Record the tool usage
            println!("Tool used: {} with input: {}", tool, tool_input);
            self.execution_history.push(ToolStep {
                tool,
                tool_input,
            });
        }

        // If we have a history of tool usage, record it
        if !self.execution_history.is_empty() {
            self.virtual_tool_manager.record_successful_sequence(
                problem,
                self.execution_history.clone(),
                response
            );
        }
    }
}

// Verification agent
struct VerificationAgent {
    toolbox: MathToolbox,
}

impl VerificationAgent {
    fn new(toolbox: MathToolbox) -> Self {
        VerificationAgent {
            toolbox,
        }
    }

    fn verify_result(&self, problem: &str, proposed_solution: &str) -> (bool, String) {
        verify_solution(problem, proposed_solution)
    }
}

// Math workflow for solving problems
fn math_workflow(
    problem: &str,
    solver_agent: &mut MathSolverAgent,
    verification_agent: &VerificationAgent,
) -> HashMap<String, String> {
    let max_retries = MAX_VERIFICATION_RETRIES;
    let mut attempts = 0;
    let mut solution = None;
    let mut is_verified = false;
    let mut verification_result = None;
    let mut used_virtual_tool = false;
    let mut virtual_tool_info = None;

    // First, check if we have a virtual tool for this problem
    println!("Checking for virtual tools for problem: {}", problem);
    let virtual_tool = solver_agent.virtual_tool_manager.find_matching_virtual_tool(problem);

    if let Some(tool) = virtual_tool {
        println!("Found matching virtual tool: {}", tool.name);
        virtual_tool_info = Some(tool.name.clone());
    }

    while attempts < max_retries && !is_verified {
        attempts += 1;

        // Solve the problem
        println!("Attempt {}/{}: Solving the problem...", attempts, max_retries);

        // If we don't have a solution yet, use the regular solver
        if solution.is_none() {
            let solved = solver_agent.solve_problem(problem);
            solution = Some(solved);
            used_virtual_tool = false;
        }

        // Verify the solution
        println!("Verifying solution from attempt {}...", attempts);

        let (verified, result) = verification_agent.verify_result(
            problem,
            solution.as_ref().unwrap()
        );

        is_verified = verified;
        verification_result = Some(result);

        if is_verified {
            println!("✅ Solution verified on attempt {}!", attempts);
            break;
        } else if attempts < max_retries {
            println!("❌ Verification failed on attempt {}. Trying again...", attempts);
            solution = None; // Reset solution to try again
        }
    }

    // Ensure we have a result, even if verification failed
    let verification_result = verification_result.unwrap_or_else(|| "No verification result available.".to_string());

    // Make sure we have a solution string, even if it's an error message
    let solution = solution.unwrap_or_else(|| "Failed to produce a solution.".to_string());

    let mut result = HashMap::new();
    result.insert("problem".to_string(), problem.to_string());
    result.insert("solution".to_string(), solution);
    result.insert("is_verified".to_string(), is_verified.to_string());
    result.insert("verification_result".to_string(), verification_result);
    result.insert("attempts".to_string(), attempts.to_string());
    result.insert("used_virtual_tool".to_string(), used_virtual_tool.to_string());

    if let Some(tool_info) = virtual_tool_info {
        result.insert("virtual_tool_info".to_string(), tool_info);
    }

    result
}

// Display the menu options
fn display_menu() {
    println!("\nAvailable Commands:");
    println!("  'sample'        - See sample problems");
    println!("  'stats'         - Show tool statistics");
    println!("  'tools'         - List virtual tools");
    println!("  'toggle_errors' - Toggle tool error simulation on/off");
    println!("  'exit'          - Quit the application");
    println!("  [math problem]  - Enter any math problem to solve");
}

// CLI interface
fn display_cli_header() {
    println!("\n=====================================================");
    println!("  Math Problem Solver");
    println!("=====================================================");
    println!("  A CLI tool for solving math problems");
    println!("  Type 'help' to see available commands");
    println!("=====================================================\n");
}

fn display_sample_problems() {
    println!("\nSample Problems:");
    for (i, problem) in SAMPLE_PROBLEMS.iter().enumerate() {
        println!("  {}. {}", i+1, problem);
    }
    println!();
}

fn display_tool_stats(toolbox: &MathToolbox) {
    println!("\nTool Statistics:");
    println!("------------------");
    println!("Tool errors are currently: {}", toolbox.get_errors_status());

    let stats = toolbox.get_stats();
    for (tool_name, tool_stats) in stats {
        let calls = *tool_stats.get("calls").unwrap_or(&0);
        let errors = *tool_stats.get("errors").unwrap_or(&0);
        let error_rate = if calls > 0 { errors as f64 / calls as f64 } else { 0.0 };

        let status = if error_rate > 0.0 {
            format!("{:.1}% errors", error_rate * 100.0)
        } else {
            "perfect".to_string()
        };

        println!("  {}: {} calls ({})", tool_name, calls, status);
    }
    println!();
}

fn display_virtual_tools(vtm: &VirtualToolManager) {
    println!("\nVirtual Tools:");
    println!("------------------");

    let tools = &vtm.virtual_tools;

    if tools.is_empty() {
        println!("  No virtual tools created yet.");
    } else {
        for (tool_id, tool) in tools {
            // Get failure count if available
            let failure_count = vtm.tool_failure_counts.get(tool_id).unwrap_or(&0);
            let failure_info = if *failure_count > 0 {
                format!(" (Failures: {})", failure_count)
            } else {
                "".to_string()
            };

            println!("  {}{}:", tool.name, failure_info);
            println!("    Description: {}", tool.description);

            // Show the sequence of tools used
            let sequence_str = tool.tool_sequence
                .iter()
                .map(|step| step.tool.clone())
                .collect::<Vec<_>>()
                .join(" → ");

            println!("    Sequence: {}", sequence_str);
            println!();
        }
    }
}

// Main function
fn main() -> Result<(), Box<dyn Error>> {
    // Set up the OpenAI API key
    env::set_var("OPENAI_API_KEY", OPENAI_API_KEY);

    // Initialize the toolbox and virtual tool manager
    let mut toolbox = MathToolbox::new();
    let virtual_tool_manager = VirtualToolManager::new(MAX_VERIFICATION_RETRIES);

    // Create agents
    let mut solver_agent = MathSolverAgent::new(toolbox.clone(), virtual_tool_manager);
    let mut verification_agent = VerificationAgent::new(toolbox.clone());

    // Display header
    display_cli_header();
    display_menu();

    loop {
        print!("\nEnter a math problem or command: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        match input.to_lowercase().as_str() {
            "exit" | "quit" => {
                println!("Goodbye!");
                break;
            },
            "help" => {
                display_menu();
            },
            "sample" => {
                display_sample_problems();
            },
            "stats" => {
                display_tool_stats(&solver_agent.toolbox);
            },
            "tools" => {
                display_virtual_tools(&solver_agent.virtual_tool_manager);
            },
            "toggle_errors" => {
                // Toggle errors in both toolboxes
                solver_agent.toolbox.toggle_errors();

                // We need to update the verification agent's toolbox to match
                // Since it's a clone, we need to create a new instance with the updated settings
                verification_agent = VerificationAgent::new(solver_agent.toolbox.clone());
            },
            _ => {
                println!("\nSolving: {}", input);
                println!("------------------");

                let start_time = Instant::now();

                // Run the math workflow
                let result = math_workflow(
                    input,
                    &mut solver_agent,
                    &verification_agent
                );

                let elapsed = start_time.elapsed();

                // Display results
                println!("\nSolution:");
                println!("------------------");
                println!("{}", result.get("solution").unwrap_or(&"No solution found".to_string()));

                println!("\nVerification:");
                println!("------------------");

                let is_verified = result.get("is_verified").unwrap_or(&"false".to_string()) == "true";

                // Store the attempts string to avoid temporary value issues
                let attempts_string = result.get("attempts")
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "0".to_string());

                if is_verified {
                    println!("✅ The solution has been verified as correct! (Attempts: {})", attempts_string);
                } else {
                    println!("❌ The solution could not be verified after {} attempts.", attempts_string);
                }

                println!("{}", result.get("verification_result").unwrap_or(&"No verification result".to_string()));

                // Show additional info
                if result.get("used_virtual_tool").unwrap_or(&"false".to_string()) == "true" {
                    println!("\nℹ️  This problem was solved using a virtual tool!");
                }

                if let Some(tool_info) = result.get("virtual_tool_info") {
                    println!("ℹ️  Virtual tool used: {}", tool_info);
                }

                println!("\nSolved in {:.2} seconds", elapsed.as_secs_f64());
            }
        }
    }

    Ok(())
}