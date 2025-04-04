pip install maturin
cd core
maturin build --release
pip install target/wheels/*.whl