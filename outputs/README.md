# Analytical Outputs & Deliverables

This directory contains precomputed benchmarks, notebook-generated datasets, visualization figures, and retail recommendation summaries.

## File Manifest

| File | Type | Description |
|---|---|---|
| `project_summary.json` | JSON | Key dataset metrics (26,878 rows, 5,000 baskets, 60 SKUs, 256 itemsets, 980 rules). |
| `top_products_by_support.csv` | CSV | SKU ranking by individual basket occurrence frequency. |
| `frequent_itemsets_apriori.csv` | CSV | 256 frequent itemsets mined via the Apriori algorithm (`min_support = 0.02`). |
| `frequent_itemsets_fpgrowth.csv` | CSV | 256 frequent itemsets mined via the FP-Growth algorithm (`min_support = 0.02`). |
| `association_rules.csv` | CSV | All 980 mined association rules with support, confidence, lift, leverage, and conviction. |
| `actionable_rules.csv` | CSV | 42 high-confidence, high-support rules passing the `support > 5%` and `lift > 2.0` business filter. |
| `planted_pattern_validation.csv` | CSV | Direct validation table of the 6 known business combos against expected benchmarks (6/6 PASS). |
| `business_recommendations.txt` | Text | Concrete operational recommendations for picking slotting, combo deals, and in-app prompts. |
| `review_notes.md` | Markdown | Comprehensive audit notes and validation confirmation. |

## Visualizations (`figures/`)

- `top_products_support.png`: Bar chart of top 10 SKUs by basket penetration.
- `rule_support_vs_lift.png`: Scatter plot showing the trade-off between rule support and lift, highlighting compound vs core patterns.
- `planted_patterns_lift.png`: Bar chart illustrating the lift strength of the six benchmark business combinations.
