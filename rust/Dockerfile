# Use Alpine Linux with Rust for static compilation
FROM rust:1.72-alpine as builder

# Install build dependencies
RUN apk add --no-cache musl-dev

# Create a new directory for our application
WORKDIR /usr/src/app

# Copy our Rust file
COPY math_solver.rs .

# Create a new empty Cargo.toml file
RUN echo '[package]' > Cargo.toml && \
    echo 'name = "math_solver"' >> Cargo.toml && \
    echo 'version = "0.1.0"' >> Cargo.toml && \
    echo 'edition = "2021"' >> Cargo.toml && \
    echo '' >> Cargo.toml && \
    echo '[dependencies]' >> Cargo.toml && \
    echo 'rand = "0.8.5"' >> Cargo.toml && \
    echo 'regex = "1.9.1"' >> Cargo.toml && \
    echo 'serde = { version = "1.0.183", features = ["derive"] }' >> Cargo.toml && \
    echo 'serde_json = "1.0.104"' >> Cargo.toml && \
    echo 'chrono = "0.4.26"' >> Cargo.toml

# Create a src directory and move the Rust file
RUN mkdir src && \
    mv math_solver.rs src/main.rs

# Build with the musl target to create a fully static binary
RUN rustup target add x86_64-unknown-linux-musl && \
    cargo build --release --target x86_64-unknown-linux-musl

# Use a minimal Alpine image for the final stage
FROM alpine:latest

# Copy the statically compiled binary
COPY --from=builder /usr/src/app/target/x86_64-unknown-linux-musl/release/math_solver /usr/local/bin/math_solver

# Set the entrypoint to our application
ENTRYPOINT ["/usr/local/bin/math_solver"]