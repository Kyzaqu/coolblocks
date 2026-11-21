"""Run the project's four evaluations (define them in docs/evaluation_plan.md).

1. Accuracy vs baselines   2. Spatial generalization   3. Temporal generalization
4. Stakeholder / usability evaluation (qualitative, see docs/stakeholder_interviews.md)
"""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
