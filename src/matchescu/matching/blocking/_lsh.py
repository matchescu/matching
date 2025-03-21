from datasketch import MinHash, MinHashLSH


class LSHBlocking:
    def __init__(self, num_perm=128, threshold=0.5, bands=16):
        self.num_perm = num_perm  # Number of permutations for MinHash
        self.threshold = threshold  # Similarity threshold for LSH
        self.bands = bands  # Number of bands for LSH bucketing
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.entities = {}  # Store entity mappings

    def _compute_minhash(self, tokens):
        """Compute MinHash signature for a set of tokens."""
        m = MinHash(num_perm=self.num_perm)
        for token in tokens:
            m.update(token.encode("utf8"))
        return m

    def add_entity(self, entity_id, tokens):
        """Add an entity with tokenized attributes to LSH."""
        minhash = self._compute_minhash(tokens)
        self.lsh.insert(entity_id, minhash)
        self.entities[entity_id] = tokens  # Store for reference

    def get_candidate_pairs(self):
        """Find candidate pairs that fall into the same LSH bucket."""
        candidate_pairs = set()
        for entity_id in self.entities:
            minhash = self._compute_minhash(self.entities[entity_id])
            neighbors = self.lsh.query(minhash)
            for neighbor in neighbors:
                if entity_id < neighbor:  # Ensure unique pairs (A, B) where A < B
                    candidate_pairs.add((entity_id, neighbor))
        return candidate_pairs


# Example usage:
lsh_blocker = LSHBlocking(num_perm=128, threshold=0.5, bands=16)
lsh_blocker.add_entity("E1", {"alice", "data", "scientist"})
lsh_blocker.add_entity("E2", {"alice", "machine", "learning"})
lsh_blocker.add_entity("E3", {"bob", "software", "developer"})
lsh_blocker.add_entity("E4", {"alice", "data", "scientist"})

print("Candidate Pairs:", lsh_blocker.get_candidate_pairs())
