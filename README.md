# Machine Learning for Neuroscience

**Graduate Course Syllabus — 15 Weeks**

*A Unified Motor Control Thread: From Single-Joint Simulation to Neural Decoding*

---

This course introduces machine learning methods commonly used in neuroscience research, with all implementations in Python. Students build foundational programming skills (Weeks 1–3), construct a multi-joint reaching dataset (Week 4), then apply 10 different ML algorithms to the same progressively enriched data — developing deep intuition for when and why each method succeeds or fails on real motor control problems.

A distinctive feature of this course is the integration of modern AI tools into the scientific workflow. Students learn to use AI assistants to accelerate data analysis, debug code, generate visualisations, and iterate on ML pipelines — skills that are rapidly becoming essential in computational neuroscience.

![Course Overview](figures/course_overview.svg)

## Course Structure

| Part | Weeks | Topic |
|------|-------|-------|
| **I. Python Foundations** | 1–3 | NumPy & Matplotlib, Pandas & Seaborn, OOP |
| **II. Course Dataset** | 4 | Multi-joint reaching, muscle synergies & PCA |
| **III. Model Selection** | 5 | Logistic regression & model evaluation |
| **IV. Supervised Learning** | 6–11 | GLMs, Naive Bayes, KNN, SVM, LDA, Random Forests |
| **V. Unsupervised Learning** | 12 | K-means clustering & GMMs |
| **VI. Neural Networks & Integration** | 13–15 | MLP concepts, temporal features, course integration & ethics |

## The Dataset

All methods from Week 5 onward are applied to the same reaching dataset:

- **480 trials** — 8 reach directions × 3 speeds × 20 subjects
- **20 subjects** — 10 healthy, 10 impaired (simulated stroke with synergy merging)
- **6 muscles** — EMG peak envelope amplitudes (biceps, triceps, anterior/posterior deltoid, pectoralis, latissimus)
- **80 neurons** — Simulated cosine-tuned M1 population (added in Week 6)
- **4,000 temporal features** — 80 neurons × 50 time bins, 10 ms resolution (added in Week 14)

Two classification tasks run throughout the course:
1. **Direction decoding** (8-class) — which of 8 directions is the subject reaching toward?
2. **Impairment detection** (binary) — is this subject healthy or a stroke survivor?

## Materials

- 📄 [**Syllabus**](syllabus/ML_Neuroscience_Syllabus.docx)
- 📖 [**Lecture PDFs**](lectures/) (Weeks 1–15)
- 📑 [**Deep Learning Handout**](handouts/Week15_Handout_Deep_Learning.pdf) — RNNs, CNNs, autoencoders, transformers with PyTorch code

## Lab Notebooks

Click the Colab badge to open any notebook directly in Google Colab. No installation required — just a Google account.

> **First time?** Run the data download cell at the top of each notebook before anything else.


| Week | Topic | Open in Colab |
|------|-------|:---:|
| 1 | Python, NumPy & Matplotlib | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tarkeshsingh/ml-neuroscience-course/blob/main/labs/Week01_Lab.ipynb) |
| 2 | Pandas & Seaborn | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tarkeshsingh/ml-neuroscience-course/blob/main/labs/Week02_Lab.ipynb) |
| 3 | Object-Oriented Programming | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tarkeshsingh/ml-neuroscience-course/blob/main/labs/Week03_Lab.ipynb) |
| 4 | Muscle Synergies & PCA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tarkeshsingh/ml-neuroscience-course/blob/main/labs/Week04_Lab.ipynb) |
| 5 | Logistic Regression & Evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tarkeshsingh/ml-neuroscience-course/blob/main/labs/Week05_Lab.ipynb) |
| 6 | GLMs — Muscles to Neurons | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tarkeshsingh/ml-neuroscience-course/blob/main/labs/Week06_Lab.ipynb) |
| 7 | Naive Bayes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tarkeshsingh/ml-neuroscience-course/blob/main/labs/Week07_Lab.ipynb) |
| 8 | K-Nearest Neighbors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tarkeshsingh/ml-neuroscience-course/blob/main/labs/Week08_Lab.ipynb) |
| 9 | Support Vector Machines | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tarkeshsingh/ml-neuroscience-course/blob/main/labs/Week09_Lab.ipynb) |
| 10 | Linear Discriminant Analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tarkeshsingh/ml-neuroscience-course/blob/main/labs/Week10_Lab.ipynb) |
| 11 | Random Forests | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tarkeshsingh/ml-neuroscience-course/blob/main/labs/Week11_Lab.ipynb) |
| 12 | K-Means & GMMs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tarkeshsingh/ml-neuroscience-course/blob/main/labs/Week12_Lab.ipynb) |
| 13 | Neural Networks — Concepts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tarkeshsingh/ml-neuroscience-course/blob/main/labs/Week13_Lab.ipynb) |
| 14 | Temporal Features | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tarkeshsingh/ml-neuroscience-course/blob/main/labs/Week14_Lab.ipynb) |


## Data Files

The course uses four progressively enriched data packages. Each is a superset of the previous:

| File | Size | What it adds | Used in |
|------|------|-------------|---------|
| `week4_data.pkl` | 54 KB | EMG features (480×6), labels, targets, subjects, fitted PCA & scaler | Weeks 4–5 |
| `week6_data.pkl` | 356 KB | + 80 neural rates (480×80), neuron properties (PDs, baselines, depths) | Weeks 6–7 |
| `week8_data.pkl` | 363 KB | + moons demo data (300×2), joint angles (480×2); drops fitted PCA/scaler | Weeks 8–13 |
| `week14_data.pkl` | 15 MB | + neural timeseries (480×4,000) — 80 neurons × 50 time bins | Week 14 |

### Loading data in Colab

Each notebook includes a data download cell at the top. If running locally, place the `.pkl` files in the same directory as the notebook.

```python
# This cell is already in each notebook — just run it
# Change the filename to match the week (week4, week6, week8, or week14)
import os, urllib.request
if not os.path.exists('week8_data.pkl'):
    base = 'https://github.com/tarkeshsingh/ml-neuroscience-course/raw/main/data/'
    for f in ['week8_data.pkl']:
        urllib.request.urlretrieve(base + f, f)
        print(f'Downloaded {f}')
```

## The 10-Method Comparison

Every method is evaluated on the same dataset with leave-one-subject-out cross-validation. The running comparison table grows each week:

| Method | Week | Neural 8-Dir | EMG Binary | Learns Features? |
|--------|------|:---:|:---:|:---:|
| LR | 5 | 91.2% | 85.0% | No |
| Naive Bayes | 7 | 95.2% | 65.4% | No |
| KNN | 8 | 94.6% | 100% | No |
| Linear SVM | 9 | 92.7% | 70.0% | No |
| RBF SVM | 9 | 94.6% | 100% | No |
| LDA | 10 | 95.4% | 78.5% | No |
| Random Forest | 11 | 95.2% | 100% | No |
| K-means | 12 | ARI=0.92 | — | No |
| GMM | 12 | ARI=0.91 | — | No |
| MLP | 13 | 94.2% | 100% | Yes |

**Week 14 lesson:** LR on temporal features (99.0%) beats MLP on mean rates (79.8%) — features matter more than the classifier.

## Requirements

The notebooks run in Google Colab with no setup. For local use:

```bash
pip install numpy matplotlib scikit-learn pandas seaborn pdfplumber
pip install torch  # Weeks 13–14 only
```

## Key Readings

| Week | Paper |
|------|-------|
| 4 | Tresch, Cheung & d'Avella (2006). Matrix factorization for muscle synergies. *J Neurophysiol* |
| 5 | Glaser et al. (2019). Supervised ML in systems neuroscience. *Prog Neurobiol* |
| 6 | Georgopoulos et al. (1982). Direction and arm movements in motor cortex. *J Neurosci* |
| 9 | Wolpaw et al. (2002). Brain–computer interfaces. *Clin Neurophysiol* |
| 10 | Eluchans et al. (2025). Eye–hand coarticulation and hierarchical planning. *J Neurophysiol* |
| 12 | Gallego et al. (2017). Neural manifolds for movement control. *Neuron* |
| 14 | Pandarinath et al. (2018). LFADS for neural population dynamics. *Nat Methods* |
| 15 | Obermeyer et al. (2019). Racial bias in healthcare algorithms. *Science* |

Full reading list with annotations is in the [syllabus](syllabus/ML_Neuroscience_Syllabus.docx).

## License

[MIT](LICENSE)

## Citation

If you use these materials in your teaching or research, please cite:

> Tarkesh Singh. *Machine Learning for Neuroscience: Graduate Course Materials.* GitHub, 2026. https://github.com/tarkeshsingh/ml-neuroscience-course
