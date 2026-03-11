# Dataset for Part B: Multiple Incremental Decremental SVM

## Dataset Used

**Dataset:** Synthetic binary classification data generated using `sklearn.datasets.make_classification`

## How the Dataset is Obtained

The dataset is generated programmatically in the notebooks using:

```python
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=300, n_features=4, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)
# Convert labels from {0,1} to {-1,+1} for SVM
y = 2 * y - 1
```

## Usage in Notebooks

- **Task 2.1, 2.2, 2.3:** Dataset is generated and used for reproducing the incremental-decremental SVM algorithm. The dataset is split into initial training set and points to be added incrementally.
- **Task 3.1, 3.2:** Same dataset generation is used for ablation studies and failure mode analysis.

## Justification

- **Reason:** The paper uses synthetic binary classification (Gaussian mixture) and real-world classification (Fisher river). Our synthetic dataset is appropriate because: (1) it has binary labels and continuous features suitable for SVM, (2) it allows controlled experiments for incremental/decremental updates, (3) it meets the requirement of ≥100 samples and ≥2 features.
- **Limitation:** Smaller than the paper's artificial data (n=500) and lacks the time-series structure of the Fisher river dataset.
