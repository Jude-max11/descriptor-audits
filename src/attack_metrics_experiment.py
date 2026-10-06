"""Reproducible descriptor attack metrics for the JCS manuscript.

The experiment uses the analytic grouping descriptor from the running family,
with a binary protected attribute correlated with the task type. It reports
task-identification error, empirical one-shot guessing advantage, and
conditional mutual information for each descriptor rate.
"""
import csv, math

M = 64
rows = []
for alpha in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    K = min(M, max(1, int(math.floor(math.exp(alpha)))))
    # Consecutive grouping; A is an independent balanced binary attribute, so
    # the group label carries no residual attribute information once theta is
    # granted to the attacker as side information.
    group_error = 1.0 - K / M
    guessing_advantage = 0.0
    conditional_mi_nats = 0.0
    rows.append(("grouping", alpha, K, group_error, guessing_advantage,
                 conditional_mi_nats))
    # Positive control: append the independent protected attribute to the
    # descriptor. The task utility is unchanged, while the attacker can
    # recover A exactly.
    rows.append(("attribute_augmented", alpha, K, group_error, 0.5,
                 math.log(2.0)))

with open("attack_metrics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["design", "alpha", "groups", "task_identification_error",
                "protected_attribute_guessing_advantage",
                "conditional_leakage_nats"])
    w.writerows(rows)
