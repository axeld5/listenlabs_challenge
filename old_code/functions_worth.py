import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from scipy import stats

@dataclass
class Individual:
    """Represents a person with binary attributes"""
    attributes: np.ndarray
    index: int
    
@dataclass
class Constraints:
    """Holds the constraint information"""
    min_percentages: Dict[int, float]  # {attribute_index: min_percentage}
    max_accept: int = 1000
    max_reject: int = 20000

@dataclass
class PopulationStats:
    """Statistics about the population"""
    attribute_probs: np.ndarray  # P(xi = 1) for each attribute
    covariance_matrix: np.ndarray  # Covariance matrix for attributes

class ThompsonSampler:
    """
    Thompson Sampling-only online selection optimizer.
    Implements only the Thompson Sampling decision rule.
    """
    
    def __init__(self, 
                 constraints: Constraints,
                 population_stats: PopulationStats):
        self.constraints = constraints
        self.stats = population_stats
        self.n_attributes = len(population_stats.attribute_probs)
        
        # State tracking
        self.accepted: List[Individual] = []
        self.rejected_count: int = 0
        self.current_counts = np.zeros(self.n_attributes)
        
        # Constraint-awareness
        self.constraint_tightness = np.ones(self.n_attributes)
        
    def reset(self):
        """Reset the optimizer state for a new run"""
        self.accepted = []
        self.rejected_count = 0
        self.current_counts = np.zeros(self.n_attributes)
        self.constraint_tightness = np.ones(self.n_attributes)
        
    def _estimate_remaining_needed(self, remaining_slots: int) -> np.ndarray:
        """
        Estimate how many more individuals with each attribute we need
        to meet constraints.
        """
        needed = np.zeros(self.n_attributes)
        n_accepted = len(self.accepted)
        
        for attr_idx, min_pct in self.constraints.min_percentages.items():
            target_count = min_pct * self.constraints.max_accept
            current_pct = self.current_counts[attr_idx] / max(n_accepted, 1)
            
            # Calculate how many more we need
            needed[attr_idx] = max(0, target_count - self.current_counts[attr_idx])
            
            # Update constraint tightness based on progress
            if n_accepted > 0:
                gap = min_pct - current_pct
                self.constraint_tightness[attr_idx] = 1 + max(0, gap * 10)
        
        return needed
    
    def _calculate_individual_score(self, individual: Individual, 
                                    remaining_slots: int) -> float:
        """Score an individual based on how well they help meet constraints."""
        needed = self._estimate_remaining_needed(remaining_slots)
        
        # Base score from attribute contribution
        score = 0.0
        for attr_idx in range(self.n_attributes):
            if individual.attributes[attr_idx] == 1:
                if attr_idx in self.constraints.min_percentages:
                    # Weighted by how much we need this attribute
                    weight = self.constraint_tightness[attr_idx]
                    score += weight * (needed[attr_idx] / max(remaining_slots, 1))
        
        # Penalty for being close to rejection limit
        rejection_pressure = self.rejected_count / self.constraints.max_reject
        score *= (1 - rejection_pressure * 0.5)
        
        # Bonus for rare valuable attributes
        for attr_idx in range(self.n_attributes):
            if individual.attributes[attr_idx] == 1:
                rarity = 1 - self.stats.attribute_probs[attr_idx]
                if attr_idx in self.constraints.min_percentages:
                    score += rarity * 0.2
        
        return score
    
    def _thompson_sampling_decision(self, individual: Individual) -> bool:
        """Make decision using Thompson Sampling strategy."""
        remaining_slots = self.constraints.max_accept - len(self.accepted)
        remaining_rejections = self.constraints.max_reject - self.rejected_count
        
        if remaining_slots == 0:
            return False
        if remaining_rejections == 0:
            return True
        
        score = self._calculate_individual_score(individual, remaining_slots)
        
        # Sample from posterior
        alpha = 1 + len(self.accepted)
        beta = 1 + self.rejected_count
        sampled_threshold = np.random.beta(alpha, beta)
        
        return score > sampled_threshold
    
    def decide(self, individual: Individual) -> bool:
        """Decide whether to accept or reject an individual."""
        decision = self._thompson_sampling_decision(individual)
        
        # Update state
        if decision:
            self.accepted.append(individual)
            self.current_counts += individual.attributes
        else:
            self.rejected_count += 1
        
        return decision
    
    def _check_constraint_progress(self) -> float:
        """Check how well we're meeting constraints (0 to 1)."""
        if len(self.accepted) == 0:
            return 1.0
        
        satisfaction_scores = []
        for attr_idx, min_pct in self.constraints.min_percentages.items():
            current_pct = self.current_counts[attr_idx] / len(self.accepted)
            satisfaction = min(1.0, current_pct / min_pct)
            satisfaction_scores.append(satisfaction)
        
        return np.mean(satisfaction_scores) if satisfaction_scores else 1.0
    
    def get_final_statistics(self) -> Dict:
        """Get final statistics about the selection process."""
        stats_dict = {
            'n_accepted': len(self.accepted),
            'n_rejected': self.rejected_count,
            'constraint_satisfaction': {},
            'attribute_percentages': {}
        }
        
        if len(self.accepted) > 0:
            for attr_idx, min_pct in self.constraints.min_percentages.items():
                actual_pct = self.current_counts[attr_idx] / len(self.accepted)
                stats_dict['attribute_percentages'][f'attr_{attr_idx}'] = actual_pct
                stats_dict['constraint_satisfaction'][f'attr_{attr_idx}'] = actual_pct >= min_pct
        
        return stats_dict