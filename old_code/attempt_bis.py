import numpy as np
from typing import Dict

from attempt import (
    Constraints,
    PopulationStats,
    PopulationGenerator,
    ThompsonSampler,
    run_simulation,
)


def run_thompson_sampling_simulations(
    constraints: Constraints,
    population_stats: PopulationStats,
    n_simulations: int = 100,
    n_individuals: int = 25000,
    verbose_first: bool = True,
) -> Dict:
    """
    Run multiple simulations using ONLY Thompson Sampling and return aggregated statistics.
    """
    generator = PopulationGenerator(population_stats)

    results = []
    for sim in range(n_simulations):
        optimizer = ThompsonSampler(constraints, population_stats)
        result = run_simulation(
            optimizer,
            generator,
            n_individuals,
            verbose=(verbose_first and sim == 0),
        )
        results.append(result)

    rejections = [r["n_rejected"] for r in results]
    accepts = [r["n_accepted"] for r in results]

    aggregated = {
        "strategy": "thompson_sampling",
        "mean_rejections": float(np.mean(rejections)),
        "std_rejections": float(np.std(rejections)),
        "min_rejections": int(np.min(rejections)),
        "max_rejections": int(np.max(rejections)),
        "mean_accepts": float(np.mean(accepts)),
        "constraint_violations": int(
            sum(1 for r in results if not all(r["constraint_satisfaction"].values()))
        ),
    }

    return aggregated


if __name__ == "__main__":
    # Define constraints
    constraints = Constraints(
        min_percentages={
            0: 0.54,  # At least 54% must have attribute 0
            1: 0.30,  # At least 30% must have attribute 1
            2: 0.45,  # At least 45% must have attribute 2
        },
        max_accept=1000,
        max_reject=20000,
    )

    # Define population statistics (example)
    n_attributes = 5
    attribute_probs = np.array([0.6, 0.35, 0.5, 0.7, 0.25])

    # Create a correlation structure
    covariance_matrix = np.eye(n_attributes) * 0.25
    for i in range(n_attributes):
        for j in range(n_attributes):
            if i != j:
                covariance_matrix[i, j] = 0.1 * np.exp(-abs(i - j))

    population_stats = PopulationStats(
        attribute_probs=attribute_probs, covariance_matrix=covariance_matrix
    )

    print("Running Thompson Sampling-only simulations:\n")
    print("-" * 60)

    results = run_thompson_sampling_simulations(
        constraints,
        population_stats,
        n_simulations=100,
        n_individuals=25000,
        verbose_first=True,
    )

    print("\nStrategy: thompson_sampling")
    print(
        f"Mean rejections: {results['mean_rejections']:.1f} ± {results['std_rejections']:.1f}"
    )
    print(
        f"Min/Max rejections: {results['min_rejections']:.0f} / {results['max_rejections']:.0f}"
    )
    print(f"Mean accepts: {results['mean_accepts']:.1f}")
    print(f"Constraint violations: {results['constraint_violations']}/100")

    print("\n" + "-" * 60)
    print("\nThompsonSampler-only runner completed.")


