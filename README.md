# Nigerian SME Vulnerability Clustering

Unsupervised clustering on a synthetic dataset of 2 million Nigerian SME loan records, used to group SMEs into three distinct financial profiles and explore what each profile means for how lenders could serve them.

## Why this project exists

A credit score tells you whether an SME is likely to default. It does not tell you why, or what kind of business sits behind that score. A retail trader and a tech consultancy can carry the same credit score and the same loan amount while having completely different financial realities.

This project asks a different question than a typical credit model. Instead of "will this SME default", it asks "what kind of business is this, and what does that mean for how we should serve it". The approach is deliberately unsupervised. No default labels were used during clustering, so the financial structure of these businesses was left to speak for itself rather than being anchored to a single outcome variable.

## Dataset

Source: [electricsheepafrica/africa-synth-banking-sme-loans-nigeria](https://huggingface.co/datasets/electricsheepafrica/africa-synth-banking-sme-loans-nigeria) via Hugging Face

The dataset is synthetic, built to reflect the structure and patterns of the Nigerian banking market between 2021 and 2024. It contains loan level records with fields covering principal amounts, revenue, collateral values, employee counts, interest rates, tenor, credit scores, and default flags across multiple business sectors.

## Method

### 1. Feature engineering

Five features were engineered to capture dimensions of financial health that raw loan data does not show on its own.

| Feature | Formula | What it captures |
|---|---|---|
| Loan to revenue ratio | Principal ÷ Annual Revenue | Borrowing appetite vs repayment capacity |
| Collateral coverage ratio | Collateral ÷ Principal | How well the loan is actually secured |
| Monthly repayment estimate | Principal ÷ Tenor | Monthly cash outflow in naira |
| Revenue per employee | Annual Revenue ÷ Employees | Business productivity and operational health |
| Debt service ratio | Monthly Repayment ÷ Monthly Revenue | Repayment pressure as a share of income |

### 2. Exploratory analysis

Key patterns found before any clustering was run:

- Retail trade and services together account for over 55% of all loans in the dataset
- Over 800,000 loans carry a 12 month tenor, by far the most common term
- Collateral value correlates with loan principal at roughly 0.97, suggesting collateral is often set as a fixed proportion of the loan rather than assessed independently against risk
- Employee counts peak sharply between 6 and 10 staff, confirming these are genuinely small businesses
- Credit scores cluster mostly between 550 and 700, with smaller populations in the tails below 450 and above 750

The correlation matrix also guided feature selection. Three pairs of features were correlated above 0.85, so the engineered ratio was kept over the raw column in each case: debt service ratio over loan to revenue ratio, collateral coverage ratio over raw collateral value, and revenue per employee over raw annual revenue.

### 3. Preprocessing

Three steps were applied before clustering, each addressing a specific problem in the raw data.

1. Scaling, so a feature like principal in naira did not dominate a feature like debt service ratio just because of its size
2. Outlier capping, to limit the influence of a handful of unusually large loans or revenue figures
3. Variance based feature selection, dropping features with variance below 0.4 that were too uniform to meaningfully separate one SME from another

### 4. Clustering

K-means was run across k values from 2 to 14. Both the elbow curve and the silhouette score curve pointed to k=3 as the optimal number of clusters.

To confirm the final feature set was actually contributing to the separation rather than adding noise, silhouette scores were compared on a 100,000 row sample:

- Full nine feature model: 0.1907
- Three core features only (principal, revenue per employee, debt service ratio): 0.4628
- Core three plus number of employees: 0.3568, a drop from the top three alone

The three core features were kept because they produced the cleanest, best supported separation. Collateral coverage ratio, interest rate, and credit score were sitting at essentially zero variance across clusters, meaning they were not doing any work to separate the groups.

## The three clusters

| Cluster | Count | Profile |
|---|---|---|
| Ordinary | 1,103,449 | Moderate loan sizes, average revenue productivity, manageable debt service. The largest and most unremarkable group. |
| Vulnerable | 459,727 | High debt service ratio relative to revenue, low revenue per employee. Often looks serviceable on paper but financially stretched. |
| Safe | 436,824 | Strong revenue per employee, low debt service pressure. The most creditworthy and most underserved group. |

## Repository contents

- `Nigerian_SME_Vulnerability_Clustering.ipynb`: full analysis, from feature engineering through clustering and cluster profiling
- Supporting chart images referenced in the writeup

## Read the full writeup

A full narrative writeup of this project, including the reasoning behind each step and directional product ideas for each cluster, is available here: [Unsupervised ML: Understanding the SME Customer Behind the Credit Score in Nigeria](https://medium.com/@ottneel/unsupervised-ml-understanding-the-sme-customer-behind-the-credit-score-b53b375eadda)

## Caveats

This dataset is synthetic and built to represent the structure of the Nigerian SME lending market, not actual records from a specific bank. The product ideas discussed in the writeup are directional starting points for discussion, not finished recommendations. They were generated from cluster level patterns and do not account for a real institution's risk appetite, regulatory constraints, or cost structure.
