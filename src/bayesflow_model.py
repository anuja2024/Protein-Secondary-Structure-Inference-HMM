import bayesflow as bf

from simulator import N_AMINO_ACIDS


def build_adapter() -> bf.adapters.Adapter:
    return (
        bf.adapters.Adapter()
        .to_array()
        .one_hot("aa_window", num_classes=N_AMINO_ACIDS)
        .convert_dtype("float64", "float32", include="target_probs")
        .rename("aa_window", "summary_variables")
        .rename("target_probs", "inference_variables")
    )


def build_approximator(summary_dim: int = 16, recurrent_dim: int = 32) -> bf.approximators.ScoringRuleApproximator:
    adapter = build_adapter()

    summary_net = bf.networks.TimeSeriesNetwork(
        summary_dim=summary_dim,
        recurrent_type="lstm",
        bidirectional=True,
        recurrent_dim=recurrent_dim,
    )

    inference_net = bf.networks.ScoringRuleNetwork(
        scoring_rules={"probs": bf.scoring_rules.CrossEntropyScore()}
    )

    approximator = bf.approximators.ScoringRuleApproximator(
        inference_network=inference_net,
        summary_network=summary_net,
        adapter=adapter,
        standardize=None,
    )
    return approximator


if __name__ == "__main__":
    approximator = build_approximator()
    print("Approximator built successfully:", type(approximator).__name__)