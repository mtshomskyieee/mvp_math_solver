echo "Tribunal"
# Example request to the tribunal endpoint
curl -X POST "http://localhost:8000/api/tribunal" \
     -H "Content-Type: application/json" \
     -d '{"problem": "What is 25 + 37?"}'
echo
echo "CAS"
# Example request to the CAS agent
curl -X POST "http://localhost:8000/api/cas" \
     -H "Content-Type: application/json" \
     -d '{"problem": "What is the square root of -1"}'
echo
echo "Solver"
# Example request to the solver agent
curl -X POST "http://localhost:8000/api/solver" \
     -H "Content-Type: application/json" \
     -d '{"problem": "Multiply 13 by 7"}'

# Example request to the verification agent
echo
echo "Verification"
curl -X POST "http://localhost:8000/api/verification" \
     -H "Content-Type: application/json" \
     -d '{"problem": "What is 144 divided by 12?", "proposed_solution": "12"}'