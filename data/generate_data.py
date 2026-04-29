from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

data = []

for _ in range(1000):
    cart_size = random.randint(1, 6)
    similar_items = random.randint(0, cart_size)
    avg_price = random.uniform(10, 200)

    return_prob = (
        0.2 * cart_size +
        0.3 * similar_items -
        0.1 * (avg_price / 100)
    )

    return_prob = min(max(return_prob, 0), 1)

    data.append([cart_size, similar_items, avg_price, return_prob])

df = pd.DataFrame(data, columns=[
    "cart_size", "similar_items", "avg_price", "return_prob"
])

output_path = Path(__file__).parent / "synthetic_data.csv"
df.to_csv(output_path, index=False)

print("Synthetic data generated!")