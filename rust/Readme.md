# Rust CLI Port of Python project

- ported from the python project to rust
- runs faster as a result of it being in rust and not connecting to streamlit 
- has the same class structure (see ../Readme.md)

# Build + Run in a Container

## ADD OPENAI KEY TO math_solver.rs
- add your key to the variable OPENAI_API_KEY around line 13 of math_solver.rs
- DO THIS FIRST
- ☝️☝️☝️☝️☝️☝️

## Build
- cd to the rust directory
- `docker build . --no-cache`

## Run CLI
- `docker compose run math_solver`

## CLI Default
Type in a math problem, and it will work on the result

## CLI Menu
While the CLI is running, type the following
- `exit` or `quit` 
- `help` --> displays the menu
- `sample` --> display sample problems, that you can type in yourself
- `stats` --> display tool usage stats
- `tools` --> display list of virtual tools created
- `toggle_errors` --> turn on/off random errors in sum/product tools

