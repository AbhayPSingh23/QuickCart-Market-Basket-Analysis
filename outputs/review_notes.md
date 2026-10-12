# Review Notes & Project Validation

- **Notebook Source**: The project notebook (`notebooks/QuickCart_Market_Basket_Analysis.ipynb`) is directly integrated and verified from the author's official Google Colab project: [Open in Colab](https://colab.research.google.com/drive/11qA3YNfb4F9Rf-24mocs9IDOia9uPLc5?usp=sharing).
- **Dataset Validation**: 26,878 transaction line items across 5,000 distinct orders, 60 unique SKUs, 12 stores, covering November 1–30, 2026. Zero missing values and zero duplicate items per basket.
- **Basket Size Distribution**: Mean basket size is 5.38 distinct items (median 5, range 1–16).
- **Frequent Itemsets**: Exactly 256 frequent itemsets discovered at `min_support = 0.02` by both Apriori and FP-Growth algorithms with 0 support discrepancies.
- **Association Rules**: 980 total association rules mined with `min_lift >= 1.0`.
- **Actionable Rules**: 42 high-affinity rules meet the business criteria (`support > 0.05` and `lift > 2.0`).
- **Planted Patterns Verification**: All 6 documented business combos (Breakfast Combo, Thali Staples, Snack-Drink Combo, Ghar Breakfast, Tea-Time, Personal Care Restock) passed validation with exact metric parity against the reference specification (6/6 PASS).
- **Negative Control Check**: Unrelated pair `Banana 1kg -> Shampoo 340ml` appeared in only 11 baskets (0.22% support), failing to clear the 2.0% support threshold, confirming that mined rules reflect genuine customer co-purchase signal rather than statistical noise.
