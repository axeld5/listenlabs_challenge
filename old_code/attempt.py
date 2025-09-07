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
    
class StochasticSelectionOptimizer:
    """
    Sophisticated online selection optimizer using adaptive threshold strategies
    with lookahead estimation and constraint-aware decision making.
    """
    
    def __init__(self, 
                 constraints: Constraints,
                 population_stats: PopulationStats,
                 strategy: str = 'adaptive_threshold'):
        """
        Initialize the optimizer.
        
        Args:
            constraints: Problem constraints
            population_stats: Population statistics
            strategy: Selection strategy ('adaptive_threshold', 'ucb', 'thompson_sampling')
        """
        self.constraints = constraints
        self.stats = population_stats
        self.strategy = strategy
        self.n_attributes = len(population_stats.attribute_probs)
        
        # State tracking
        self.accepted = []
        self.rejected_count = 0
        self.current_counts = np.zeros(self.n_attributes)
        
        # Adaptive parameters
        self.acceptance_threshold = 0.5
        self.threshold_history = []
        self.constraint_tightness = np.ones(self.n_attributes)
        
    def reset(self):
        """Reset the optimizer state for a new run"""
        self.accepted = []
        self.rejected_count = 0
        self.current_counts = np.zeros(self.n_attributes)
        self.acceptance_threshold = 0.5
        self.threshold_history = []
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
        """
        Calculate a score for an individual based on how well they help
        meet constraints.
        """
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
    
    def _adaptive_threshold_decision(self, individual: Individual) -> bool:
        """
        Make decision using adaptive threshold strategy.
        """
        remaining_slots = self.constraints.max_accept - len(self.accepted)
        remaining_rejections = self.constraints.max_reject - self.rejected_count
        
        if remaining_slots == 0:
            return False
        if remaining_rejections == 0:
            return True
            
        score = self._calculate_individual_score(individual, remaining_slots)
        
        # Adaptive threshold based on progress
        progress = len(self.accepted) / self.constraints.max_accept
        
        # Check constraint satisfaction
        constraint_satisfaction = self._check_constraint_progress()
        
        # Adjust threshold based on constraint satisfaction and progress
        if constraint_satisfaction < 0.8 and progress > 0.3:
            # We're behind on constraints, be more selective
            self.acceptance_threshold *= 0.98
        elif constraint_satisfaction > 0.95 and progress < 0.7:
            # We're doing well on constraints, can be less selective
            self.acceptance_threshold *= 1.02
            
        # Clamp threshold
        self.acceptance_threshold = np.clip(self.acceptance_threshold, 0.1, 0.9)
        
        # Make decision with some randomness for exploration
        decision_prob = 1 / (1 + np.exp(-10 * (score - self.acceptance_threshold)))
        
        return np.random.random() < decision_prob
    
    def _ucb_decision(self, individual: Individual) -> bool:
        """
        Make decision using Upper Confidence Bound strategy.
        """
        remaining_slots = self.constraints.max_accept - len(self.accepted)
        remaining_rejections = self.constraints.max_reject - self.rejected_count
        
        if remaining_slots == 0:
            return False
        if remaining_rejections == 0:
            return True
            
        score = self._calculate_individual_score(individual, remaining_slots)
        
        # Add exploration bonus
        n_seen = len(self.accepted) + self.rejected_count + 1
        exploration_bonus = np.sqrt(2 * np.log(n_seen) / max(len(self.accepted), 1))
        
        ucb_score = score + exploration_bonus
        
        # Dynamic threshold based on remaining capacity
        threshold = 0.5 * (1 - remaining_slots / self.constraints.max_accept)
        
        return ucb_score > threshold
    
    def _thompson_sampling_decision(self, individual: Individual) -> bool:
        """
        Make decision using Thompson Sampling strategy.
        """
        remaining_slots = self.constraints.max_accept - len(self.accepted)
        remaining_rejections = self.constraints.max_reject - self.rejected_count
        
        if remaining_slots == 0:
            return False
        if remaining_rejections == 0:
            return True
            
        # Maintain Beta distributions for acceptance probability
        score = self._calculate_individual_score(individual, remaining_slots)
        
        # Sample from posterior
        alpha = 1 + len(self.accepted)
        beta = 1 + self.rejected_count
        sampled_threshold = np.random.beta(alpha, beta)
        
        return score > sampled_threshold
    
    def _check_constraint_progress(self) -> float:
        """
        Check how well we're meeting constraints (0 to 1).
        """
        if len(self.accepted) == 0:
            return 1.0
            
        satisfaction_scores = []
        for attr_idx, min_pct in self.constraints.min_percentages.items():
            current_pct = self.current_counts[attr_idx] / len(self.accepted)
            satisfaction = min(1.0, current_pct / min_pct)
            satisfaction_scores.append(satisfaction)
            
        return np.mean(satisfaction_scores) if satisfaction_scores else 1.0
    
    def decide(self, individual: Individual) -> bool:
        """
        Make a decision whether to accept or reject an individual.
        
        Returns:
            True if accept, False if reject
        """
        # Choose strategy
        if self.strategy == 'adaptive_threshold':
            decision = self._adaptive_threshold_decision(individual)
        elif self.strategy == 'ucb':
            decision = self._ucb_decision(individual)
        elif self.strategy == 'thompson_sampling':
            decision = self._thompson_sampling_decision(individual)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Update state
        if decision:
            self.accepted.append(individual)
            self.current_counts += individual.attributes
        else:
            self.rejected_count += 1
            
        return decision
    
    def get_final_statistics(self) -> Dict:
        """
        Get final statistics about the selection process.
        """
        stats = {
            'n_accepted': len(self.accepted),
            'n_rejected': self.rejected_count,
            'constraint_satisfaction': {},
            'attribute_percentages': {}
        }
        
        if len(self.accepted) > 0:
            for attr_idx, min_pct in self.constraints.min_percentages.items():
                actual_pct = self.current_counts[attr_idx] / len(self.accepted)
                stats['attribute_percentages'][f'attr_{attr_idx}'] = actual_pct
                stats['constraint_satisfaction'][f'attr_{attr_idx}'] = actual_pct >= min_pct
                
        return stats

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

# Common misspelling alias
ThomsonSampler = ThompsonSampler

class PopulationGenerator:
    """Generate synthetic population based on statistics"""
    
    def __init__(self, population_stats: PopulationStats):
        self.stats = population_stats
        self.n_attributes = len(population_stats.attribute_probs)
        
        # Generate correlation matrix from covariance
        self.correlation_matrix = self._cov_to_corr(population_stats.covariance_matrix)
        
    def _cov_to_corr(self, cov: np.ndarray) -> np.ndarray:
        """Convert covariance to correlation matrix"""
        D = np.sqrt(np.diag(cov))
        D_inv = np.diag(1 / D)
        return D_inv @ cov @ D_inv
    
    def generate_individual(self, index: int) -> Individual:
        """Generate a single individual with correlated binary attributes"""
        # Use Gaussian copula approach for correlated binary variables
        
        # Generate correlated normal variables
        mean = np.zeros(self.n_attributes)
        z = np.random.multivariate_normal(mean, self.correlation_matrix)
        
        # Transform to uniform
        u = stats.norm.cdf(z)
        
        # Transform to binary with desired marginal probabilities
        attributes = (u < self.stats.attribute_probs).astype(int)
        
        return Individual(attributes=attributes, index=index)
    
    def generate_stream(self, n: int) -> List[Individual]:
        """Generate a stream of n individuals"""
        return [self.generate_individual(i) for i in range(n)]

def run_simulation(optimizer: StochasticSelectionOptimizer, 
                  generator: PopulationGenerator,
                  n_individuals: int,
                  verbose: bool = False) -> Dict:
    """
    Run a single simulation of the selection process.
    """
    optimizer.reset()
    
    for i in range(n_individuals):
        individual = generator.generate_individual(i)
        decision = optimizer.decide(individual)
        
        if verbose and i % 1000 == 0:
            progress = optimizer._check_constraint_progress()
            print(f"Progress: {i}/{n_individuals}, "
                  f"Accepted: {len(optimizer.accepted)}, "
                  f"Rejected: {optimizer.rejected_count}, "
                  f"Constraint satisfaction: {progress:.2f}")
        
        # Check stopping conditions
        if len(optimizer.accepted) >= optimizer.constraints.max_accept:
            if verbose:
                print(f"Reached max accepts at individual {i}")
            break
        if optimizer.rejected_count >= optimizer.constraints.max_reject:
            if verbose:
                print(f"Reached max rejects at individual {i}")
            break
    
    return optimizer.get_final_statistics()

def run_multiple_simulations(constraints: Constraints,
                           population_stats: PopulationStats,
                           n_simulations: int = 100,
                           n_individuals: int = 25000,
                           strategy: str = 'adaptive_threshold') -> Dict:
    """
    Run multiple simulations and return aggregated statistics.
    """
    generator = PopulationGenerator(population_stats)
    
    results = []
    for sim in range(n_simulations):
        optimizer = StochasticSelectionOptimizer(constraints, population_stats, strategy)
        result = run_simulation(optimizer, generator, n_individuals, verbose=(sim == 0))
        results.append(result)
    
    # Aggregate results
    rejections = [r['n_rejected'] for r in results]
    accepts = [r['n_accepted'] for r in results]
    
    aggregated = {
        'strategy': strategy,
        'mean_rejections': np.mean(rejections),
        'std_rejections': np.std(rejections),
        'min_rejections': np.min(rejections),
        'max_rejections': np.max(rejections),
        'mean_accepts': np.mean(accepts),
        'constraint_violations': sum(1 for r in results 
                                    if not all(r['constraint_satisfaction'].values()))
    }
    
    return aggregated

# Example usage
if __name__ == "__main__":
    # Define constraints
    # Create the Thompson Sampling policy
    constraints_data = [
        {'attribute': 'techno_lover', 'minCount': 650}, 
        {'attribute': 'well_connected', 'minCount': 450}, 
        {'attribute': 'creative', 'minCount': 300}, 
        {'attribute': 'berlin_local', 'minCount': 750}
    ]

    population_data = {
        'relativeFrequencies': {
            'techno_lover': 0.6265000000000001, 
            'well_connected': 0.4700000000000001, 
            'creative': 0.06227, 
            'berlin_local': 0.398
        }, 
        'correlations': {
            'techno_lover': {'techno_lover': 1, 'well_connected': -0.4696169332674324, 'creative': 0.09463317039891586, 'berlin_local': -0.6549403815606182}, 
            'well_connected': {'techno_lover': -0.4696169332674324, 'well_connected': 1, 'creative': 0.14197259140471485, 'berlin_local': 0.5724067808436452}, 
            'creative': {'techno_lover': 0.09463317039891586, 'well_connected': 0.14197259140471485, 'creative': 1, 'berlin_local': 0.14446459505650772}, 
            'berlin_local': {'techno_lover': -0.6549403815606182, 'well_connected': 0.5724067808436452, 'creative': 0.14446459505650772, 'berlin_local': 1}
        }
    }

    constraints = Constraints(
        min_percentages={
            0: 0.65, 
            1: 0.45, 
            2: 0.30, 
            3: 0.75
        },
        max_accept=1000,
        max_reject=20000
    )
    
    # Define population statistics (example)
    n_attributes = 4
    attribute_probs = np.array([0.62, 0.47, 0.06, 0.39])
    
    # Create a correlation structure
    # Build correlation matrix from population data
    attribute_names = ['techno_lover', 'well_connected', 'creative', 'berlin_local']
    covariance_matrix = np.zeros((n_attributes, n_attributes))
    
    for i, attr_i in enumerate(attribute_names):
        for j, attr_j in enumerate(attribute_names):
            correlation = population_data['correlations'][attr_i][attr_j]
            # Convert correlation to covariance using attribute probabilities
            std_i = np.sqrt(attribute_probs[i] * (1 - attribute_probs[i]))
            std_j = np.sqrt(attribute_probs[j] * (1 - attribute_probs[j]))
            covariance_matrix[i, j] = correlation * std_i * std_j
    
    population_stats = PopulationStats(
        attribute_probs=attribute_probs,
        covariance_matrix=covariance_matrix
    )
    
    # Compare strategies
    strategies = ['adaptive_threshold', 'ucb', 'thompson_sampling']
    
    print("Comparing strategies over 100 simulations each:\n")
    print("-" * 60)
    
    for strategy in strategies:
        results = run_multiple_simulations(
            constraints, 
            population_stats,
            n_simulations=100,
            n_individuals=25000,
            strategy=strategy
        )
        
        print(f"\nStrategy: {strategy}")
        print(f"Mean rejections: {results['mean_rejections']:.1f} ± {results['std_rejections']:.1f}")
        print(f"Min/Max rejections: {results['min_rejections']:.0f} / {results['max_rejections']:.0f}")
        print(f"Mean accepts: {results['mean_accepts']:.1f}")
        print(f"Constraint violations: {results['constraint_violations']}/100")
    
    print("\n" + "-" * 60)
    print("\nOptimizer successfully created and tested!")
    print("You can adjust constraints, population statistics, and strategies as needed.")