#!/usr/bin/env python3
"""
Second Scenario - Berghain Challenge
Scenario 2: techno_lover, well_connected, creative, berlin_local attributes
"""

import numpy as np
from typing import Dict, Any, Optional

from sampler import CorrelatedBernoulli
from scenarios import SCENARIOS

SCENARIO_ID = "custom_scenario_2"


def initialize_game() -> Dict[str, Any]:
    """Initialize a new game locally using the scenario configuration."""
    scenario = SCENARIOS[SCENARIO_ID]
    rng = np.random.default_rng(seed=42)

    sampler = CorrelatedBernoulli(scenario.marginals, scenario.corr, rng)

    game_state = {
        "scenario": scenario,
        "sampler": sampler,
        "constraints": scenario.required_counts(),
        "attributeStatistics": {attr: scenario.marginals[i] for i, attr in enumerate(scenario.attrs)},
        "person_index": 0,
        "admittedCount": 0,
        "rejectedCount": 0,
        "status": "running",
        "attribute_counts": {attr: 0 for attr in scenario.attrs}  # Track counts of admitted attributes
    }

    print(f"New game initialized for {scenario.name}")
    print("Constraints:", game_state["constraints"])
    print("Statistics:", game_state["attributeStatistics"])

    return game_state


def get_next_person(game_state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the next person using the sampler."""
    scenario = game_state["scenario"]
    sampler = game_state["sampler"]

    # Sample attributes
    sample = sampler.sample()
    attributes = {attr: bool(sample[i]) for i, attr in enumerate(scenario.attrs)}

    person = {
        "personIndex": game_state["person_index"],
        "attributes": attributes
    }

    return person


def check_constraints_satisfied(game_state: Dict[str, Any]) -> bool:
    """Check if all constraints are satisfied."""
    constraints = game_state["constraints"]
    attribute_counts = game_state["attribute_counts"]

    for attr, required_count in constraints.items():
        if attribute_counts.get(attr, 0) < required_count:
            return False
    return True


def make_decision(game_state: Dict[str, Any], person_index: int, accept: bool, person_attributes: Dict[str, bool]) -> Dict[str, Any]:
    """Process the decision and update game state."""
    scenario = game_state["scenario"]

    if accept:
        game_state["admittedCount"] += 1
        # Update attribute counts for admitted person
        for attr, has_attr in person_attributes.items():
            if has_attr:
                game_state["attribute_counts"][attr] += 1
    else:
        game_state["rejectedCount"] += 1

    # Update person index for next person
    game_state["person_index"] += 1

    # Check game ending conditions
    if game_state["rejectedCount"] >= scenario.max_rejects:
        game_state["status"] = "failed"
        print(f"Game failed: exceeded maximum rejects ({scenario.max_rejects})")
    elif game_state["admittedCount"] >= scenario.N:
        game_state["status"] = "completed"
        print(f"Game completed: processed all {scenario.N} people")

    return game_state


def main():
    """Main function to run the second scenario."""
    # Initialize new game
    game_state = initialize_game()

    # Get first person
    next_person = get_next_person(game_state)

    # Initialize counters
    total_ct = 0
    total_wcb = 0
    total_b = 0
    total_t = 0
    total_c = 0
    total_wc = 0
    n_accepted = 0

    # Main game loop
    while game_state.get("status") == "running" and next_person:
        idx = next_person["personIndex"]
        attrs = next_person["attributes"]  # { attributeId: bool }
        t, wc, c, b = (
            attrs.get("techno_lover", False),
            attrs.get("well_connected", False),
            attrs.get("creative", False),
            attrs.get("berlin_local", False)
        )

        # Decision logic for scenario 2
        accept = False

        if b and t and (total_b < 750 or total_t < 650):
            accept = True
        elif c:
            accept = True
        elif b and wc:
            if total_wcb < 150 and total_wc < 450:
                accept = True
                total_wcb += 1
        elif total_t < 650 and total_b >= 750:
            if t:
                accept = True
        elif total_t >= 650 and total_b < 750:
            if b:
                accept = True
        else:
            if c:
                accept = True

        # Update counters if accepted
        if accept:
            total_b += b
            total_t += t
            total_c += c
            total_wc += wc
            n_accepted += 1

        # Make the decision
        game_state = make_decision(game_state, person_index=idx, accept=accept, person_attributes=attrs)

        # Get next person if game is still running
        if game_state.get("status") == "running":
            next_person = get_next_person(game_state)
        else:
            next_person = None

        admitted = game_state.get("admittedCount", 0)
        rejected = game_state.get("rejectedCount", 0)

        print(f"Person {idx}: accept={accept} | admitted={admitted} rejected={rejected}")

    # Game ended - check constraint satisfaction
    if game_state.get("status") == "completed":
        print("Game completed.")
        print("Final rejected count:", game_state.get("rejectedCount"))
        print("Final attribute counts:", game_state.get("attribute_counts"))
        print("Constraints satisfied:", check_constraints_satisfied(game_state))
    else:
        print("Game ended with state:", game_state)


if __name__ == "__main__":
    main()
