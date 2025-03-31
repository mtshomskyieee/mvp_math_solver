#!/bin/bash

# For development reasons, let's squash everything together

# Define the directories to traverse
directories=("config" "core" "agents" "utils" "workflows")

# Create or clear the all.py file
> all.py

cat config/*.py >> all.py
cat core/*.py >> all.py
cat agents/*.py >> all.py
cat utils/*.py >> all.py
cat workflows/*.py >> all.py
cat ui/problem_solver.py >> all.py
cat ui/sidebar.py >> all.py
cat ui/main_app.py >> all.py
cat main.py >> all.py
