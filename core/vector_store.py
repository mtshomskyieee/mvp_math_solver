# math_solver/core/vector_store.py
import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from langchain_openai import OpenAIEmbeddings
import faiss
from utils.logging_utils import setup_logger

logger = setup_logger("vector_store")


class MathProblemVectorStore:
    """Stores and retrieves math problems using vector embeddings."""

    def __init__(self, embedding_model="text-embedding-3-small"):
        """Initialize the vector store with the embedding model."""
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.dimension = 1536  # Dimension of the OpenAI embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        self.problem_map = {}  # Maps index positions to problem hashes
        self.reverse_map = {}  # Maps problem hashes to index positions

    def add_problem(self, problem: str, problem_hash: str, tool_sequence: List[Dict[str, Any]]) -> None:
        """
        Add a math problem to the vector store.

        Args:
            problem: The original math problem text
            problem_hash: The hash identifier for the problem
            tool_sequence: The sequence of tools used to solve the problem
        """
        # Create a rich representation combining the problem and tools used
        tool_names = [step['tool'] for step in tool_sequence]
        rich_repr = f"Problem: {problem} Tools: {' -> '.join(tool_names)}"

        # Get embedding
        embedding = self.embeddings.embed_query(rich_repr)
        embedding_np = np.array([embedding], dtype=np.float32)

        # Add to FAISS index
        index_position = self.index.ntotal
        self.index.add(embedding_np)

        # Update mappings
        self.problem_map[index_position] = problem_hash
        self.reverse_map[problem_hash] = index_position

        logger.info(f"Added problem to vector store: {problem}")

    def find_similar_problems(self, problem: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar problems to the given problem.

        Args:
            problem: The problem to find similar problems for
            k: The number of similar problems to return

        Returns:
            List of tuples (problem_hash, similarity_score)
        """
        # Get embedding for the query problem
        embedding = self.embeddings.embed_query(problem)
        embedding_np = np.array([embedding], dtype=np.float32)

        # Search the index
        k = min(k, self.index.ntotal)  # Ensure k is not larger than the index size
        if k == 0:
            return []

        distances, indices = self.index.search(embedding_np, k)

        # Convert to problem hashes and scores
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]

            # Convert distance to similarity score (1 / (1 + distance))
            similarity = 1 / (1 + distance)

            if idx in self.problem_map:
                problem_hash = self.problem_map[idx]
                results.append((problem_hash, similarity))

        return results

    def update_problem(self, problem_hash: str, problem: str, tool_sequence: List[Dict[str, Any]]) -> None:
        """Update an existing problem with new information."""
        # Remove old embedding if it exists
        if problem_hash in self.reverse_map:
            index_position = self.reverse_map[problem_hash]
            # Note: FAISS doesn't support direct updates, so we would need to rebuild the index
            # For simplicity, we'll just add a new embedding and let the old one become orphaned
            logger.info(f"Updating problem in vector store: {problem}")

        # Add the problem with updated information
        self.add_problem(problem, problem_hash, tool_sequence)

    def remove_problem(self, problem_hash: str) -> bool:
        """
        Remove a problem from the vector store.

        Note: This is a "soft" remove that just updates the mappings.
        A full remove would require rebuilding the FAISS index.
        """
        if problem_hash in self.reverse_map:
            index_position = self.reverse_map[problem_hash]
            del self.problem_map[index_position]
            del self.reverse_map[problem_hash]
            logger.info(f"Removed problem from vector store: {problem_hash}")
            return True
        return False

    def save(self, filepath: str = "math_problems_vector_store.faiss") -> None:
        """Save the vector store to disk."""
        import pickle

        # Save the FAISS index
        faiss.write_index(self.index, f"{filepath}.index")

        # Save the mappings
        with open(f"{filepath}.mappings", "wb") as f:
            pickle.dump((self.problem_map, self.reverse_map), f)

        logger.info(f"Saved vector store to {filepath}")

    def load(self, filepath: str = "math_problems_vector_store.faiss") -> bool:
        """Load the vector store from disk."""
        import pickle

        try:
            # Load the FAISS index
            if os.path.exists(f"{filepath}.index"):
                self.index = faiss.read_index(f"{filepath}.index")

                # Load the mappings
                with open(f"{filepath}.mappings", "rb") as f:
                    self.problem_map, self.reverse_map = pickle.load(f)

                logger.info(f"Loaded vector store from {filepath} with {self.index.ntotal} entries")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False