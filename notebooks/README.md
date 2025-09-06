# Training & Evaluation Notebooks

This directory contains Jupyter notebooks for training, evaluation, and analysis of the Foresight SAR system.

## Structure

- `training/` - Model training notebooks
- `evaluation/` - Model evaluation and performance analysis
- `analysis/` - Data analysis and visualization notebooks
- `outputs/` - Generated outputs (gitignored)

## Getting Started

1. Install Jupyter:
   ```bash
   pip install jupyter notebook jupyterlab
   ```

2. Start Jupyter Lab:
   ```bash
   jupyter lab
   ```

3. Navigate to the notebooks directory and open the desired notebook.

## Available Notebooks

### Training
- `model_training.ipynb` - Complete model training pipeline
- `data_augmentation.ipynb` - Data augmentation experiments
- `hyperparameter_tuning.ipynb` - Hyperparameter optimization

### Evaluation
- `model_evaluation.ipynb` - Model performance evaluation
- `detection_analysis.ipynb` - Detection accuracy analysis
- `benchmark_comparison.ipynb` - Model comparison benchmarks

### Analysis
- `dataset_analysis.ipynb` - Training dataset analysis
- `error_analysis.ipynb` - Error case analysis
- `performance_profiling.ipynb` - Performance profiling

## Guidelines

- Keep notebooks focused on specific tasks
- Document all experiments and findings
- Use clear cell outputs and markdown explanations
- Save important results to the `outputs/` directory
- Clean up notebook outputs before committing (outputs are gitignored)

## Dependencies

Notebooks use the same dependencies as the main project. Install additional notebook-specific packages as needed:

```bash
pip install matplotlib seaborn plotly pandas
```