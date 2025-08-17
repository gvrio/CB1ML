# CB1-SmartPred: A Novel Machine Learning Tool for Predicting CB1-liability of Synthetic Cannabinoids

**Authors:** Gurveer Singh Shienh, Aravindhan Ganesan  
**Institution:** School of Pharmacy, University of Waterloo, Waterloo, Ontario, Canada  
**Research Period:** 2022-2023

## Overview

CB1-SmartPred is a sophisticated machine learning pipeline that provides rapid and accurate prediction of CB1 receptor binding affinity and activity classification. This tool addresses the critical bottleneck in drug development where traditional computational screening methods can take days to evaluate even a single compound, while our ML-based approach can screen thousands of compounds efficiently.

## Background

The endocannabinoid system (ECS) consists of endogenous cannabinoid molecules and two main receptors: cannabinoid receptor 1 (CB1) and cannabinoid receptor 2 (CB2). The CB1 receptor is the most prominent receptor within the mammalian brain and is the primary target for psychoactive effects induced by phytocannabinoids such as Δ9-tetrahydrocannabinol (THC). Understanding CB1 liability is crucial for drug development, as unwanted CB1 activity can lead to psychoactive side effects.

Current drug screening methods are computationally expensive and time-consuming, creating significant bottlenecks in pharmaceutical research. CB1-SmartPred overcomes these limitations through advanced machine learning approaches.

## Key Features

### Dual Prediction Capabilities
- **Binary Classification**: Predicts ACTIVE vs INACTIVE compounds against CB1 receptor
- **Regression Analysis**: Quantitative prediction of binding affinity (pKi values)
- **Ensemble Modeling**: Multiple algorithms for robust predictions

### Advanced Machine Learning Pipeline
- **Bayesian Hyperparameter Optimization**: Uses Tree-structured Parzen Estimator (TPE) for efficient parameter search
- **Cross-Validation Framework**: 5-fold stratified cross-validation for reliable performance estimates  
- **Class Imbalance Handling**: SMOTE-Tomek sampling for balanced training data
- **Feature Selection**: Random Forest-based importance ranking with configurable thresholds

### Sophisticated Molecular Featurization
- **ECFP6 Fingerprints**: 2048-bit Extended-Connectivity Fingerprints capturing molecular substructures
- **RDKit Descriptors**: 111 carefully selected physicochemical and topological descriptors
- **Feature Engineering**: Advanced dimensionality reduction and selection techniques

### Robust Data Processing
- **SMILES Standardization**: Automated molecular structure normalization
- **Tanimoto Similarity Filtering**: Chemical space analysis for dataset curation
- **Quality Control**: Comprehensive data validation and error handling

## Technical Implementation

### Core Algorithms
- **Support Vector Machine (SVM)**: Hyperplane-based classification with RBF kernel optimization
- **Multilayer Perceptron (MLP)**: Neural network with configurable architecture (100-1500 hidden units)
- **Random Forest (RF)**: Ensemble method with optimized depth and estimator count

### Performance Optimization
- **Hyperparameter Space**: Comprehensive parameter grids for each algorithm
- **Cross-Validation**: Stratified k-fold with multiple random states for stability
- **Evaluation Metrics**: Accuracy, F1-score, Matthews Correlation Coefficient, G-mean
- **Feature Thresholds**: Configurable importance-based feature selection (0-100% retention)

### Dataset Characteristics
- **Training Data**: Curated from multiple chemical databases including ChEMBL
- **Molecular Diversity**: Broad chemical space coverage with Tanimoto similarity filtering
- **Class Distribution**: Balanced active/inactive compounds through intelligent sampling
- **Validation Strategy**: Held-out test sets with temporal and chemical diversity considerations

## Results

### Classification Performance
- **Best Model**: Support Vector Machine with ECFP6 features
- **Accuracy**: >98% on validation sets
- **Matthews Correlation Coefficient**: >0.92 indicating excellent predictive power
- **Feature Method**: ECFP6 fingerprints consistently outperformed RDKit descriptors

### Key Technical Achievements
- **Bayesian Optimization**: Efficient hyperparameter search reducing computational overhead
- **Feature Selection**: Median-based importance thresholding improved model generalization  
- **Cross-Validation**: Robust 5-fold validation ensuring reliable performance estimates
- **Ensemble Integration**: Multiple algorithms providing complementary predictive capabilities

### Computational Efficiency
- **Speed**: Orders of magnitude faster than traditional docking methods
- **Scalability**: Capable of screening thousands of compounds simultaneously
- **Resource Efficiency**: Optimized memory usage and processing requirements

## Repository Structure

```
├── main.py                 # Main experimental pipeline with full ML workflow
├── custom_funcs.py         # Molecular featurization and utility functions
├── build_models.py         # Model training and serialization for deployment
├── configuration.py        # Hyperparameter search spaces and optimization settings
├── data_processing.py      # Data cleaning and preprocessing workflows
├── feat_sel.py            # Feature selection and importance analysis
├── plots.py               # Visualization and results analysis
├── new_script.py          # Similarity analysis and dataset curation
└── sandbox.py             # Development and testing utilities
```

### Key Components

**main.py**: Complete ML pipeline including:
- Bayesian hyperparameter optimization using hyperopt
- Cross-validation with stratified sampling
- SMOTE-based class balancing
- Feature selection with Random Forest importance
- Comprehensive model evaluation

**custom_funcs.py**: Molecular processing utilities:
- ECFP6 fingerprint computation using RDKit
- 111 molecular descriptor calculations
- Tanimoto similarity analysis
- SMILES standardization and validation

**build_models.py**: Production model training:
- Optimized model serialization
- Deployment-ready model artifacts
- Performance validation on held-out test sets

## Installation & Requirements

```bash
# Core dependencies
pip install pandas numpy matplotlib
pip install scikit-learn
pip install rdkit-pypi
pip install hyperopt
pip install imbalanced-learn

# Optional for visualization
pip install seaborn
pip install matplotlib
```

### System Requirements
- Python 3.7+
- 8GB+ RAM recommended for large datasets
- Multi-core CPU beneficial for cross-validation

## Usage Example

```python
from custom_funcs import computeFP
from sklearn.ensemble import RandomForestClassifier

# Load and featurize molecules
smiles_list = ['CCO', 'CCN', ...]  # Your SMILES
labels = [1, 0, ...]               # Activity labels

# Generate ECFP6 features
features_df = computeFP(smiles_list, labels)

# Train model (simplified example)
model = RandomForestClassifier()
X = features_df.drop('labels', axis=1)
y = features_df['labels']
model.fit(X, y)
```

## Research Impact

### Scientific Contributions
- **Methodology**: Novel application of Bayesian optimization to molecular property prediction
- **Performance**: Achieved state-of-the-art accuracy for CB1 binding prediction
- **Efficiency**: Demonstrated significant computational advantages over traditional methods
- **Reproducibility**: Open-source implementation with comprehensive documentation

### Future Applications
- **Drug Discovery**: Integration into pharmaceutical screening pipelines
- **ADMET Prediction**: Extension to other pharmacokinetic properties
- **Multi-Target Modeling**: Expansion to CB2 and other cannabinoid receptors
- **Generative Modeling**: Foundation for novel molecule design

## Academic Context

This work represents advanced undergraduate research in computational drug discovery, demonstrating:
- **Technical Proficiency**: Implementation of sophisticated ML pipelines
- **Domain Expertise**: Understanding of medicinal chemistry and pharmacology  
- **Research Methodology**: Rigorous experimental design and validation
- **Innovation**: Novel application of modern ML techniques to pharmaceutical problems

*Note: This code represents research-grade implementation completed during undergraduate studies (2022-2023), demonstrating graduate-level technical capabilities in computational drug discovery.*

## Citation

If you use this work in your research, please cite:
```bibtex
@misc{shienh2023cb1smartpred,
  title={CB1-SmartPred: A Novel Machine Learning Tool for Predicting CB1-liability of Synthetic Cannabinoids},
  author={Shienh, Gurveer Singh and Ganesan, Aravindhan},
  year={2023},
  institution={University of Waterloo School of Pharmacy},
  note={Available at: https://github.com/gurveershienh/CB1ML}
}
```

## License

This project is released under MIT License for academic and research purposes.

## Contact

For questions about this research or potential collaborations:
- **Primary Author**: Gurveer Singh Shienh
- **Supervisor**: Dr. Aravindhan Ganesan, University of Waterloo School of Pharmacy

---

*This project demonstrates the successful application of advanced machine learning techniques to pharmaceutical research, providing both methodological innovations and practical tools for drug discovery.*