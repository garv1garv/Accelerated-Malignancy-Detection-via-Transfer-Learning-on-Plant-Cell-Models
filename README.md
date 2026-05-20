# Plant2Skin Transfer Learning Pipeline

##  **CRITICAL SAFETY DISCLAIMER** 

**THIS REPOSITORY IS FOR RESEARCH AND DEMONSTRATION PURPOSES ONLY.**

- **NOT FOR CLINICAL USE**: This code is a research prototype and should NEVER be used for actual medical diagnosis or clinical decision-making.
- **SYNTHETIC DATA ONLY**: By default, this repository uses procedurally generated synthetic images. No real patient data is included.
- **RESEARCH PROTOTYPE**: This demonstrates transfer learning concepts from plant cell imagery to skin lesion classification as a proof-of-concept.
- **REQUIRES CLINICAL VALIDATION**: Any adaptation for real medical use requires extensive clinical validation, IRB approval, and regulatory compliance.

**DO NOT USE THIS CODE FOR MEDICAL DIAGNOSIS OR CLINICAL DECISIONS.**

## Overview

This repository demonstrates a transfer learning pipeline that adapts a convolutional neural network pretrained on plant cell imagery for the downstream task of classifying skin lesion images as benign vs malignant. The project serves as a research prototype to explore domain adaptation techniques between biological imaging domains.

## Quick Start (Synthetic Data Demo)

```bash
# Install dependencies
make install
# or: pip install -r requirements.txt

# Generate synthetic dataset and dummy checkpoint
make prepare
make make_checkpoint

# Run complete pipeline
make all
```

This will:
1. Generate synthetic skin-like images (120 total: 60 benign, 60 malignant)
2. Create a dummy "plant cell" pretrained checkpoint
3. Train a transfer learning model
4. Evaluate performance
5. Generate Grad-CAM explanations

## Project Structure

```
plant2skin-transfer/
├── data/synthetic/          # Generated synthetic dataset
├── checkpoints/             # Model checkpoints
├── src/                     # Core implementation
├── scripts/                 # Data generation and utility scripts
├── notebooks/               # Demo notebooks
├── tests/                   # Unit tests
├── outputs/                 # Results and visualizations
└── ETHICS_AND_LIMITATIONS.md # Safety and ethical considerations
```

## Transfer Learning Strategy

The pipeline implements the following transfer learning approach:

1. **Source Domain**: Plant cell imagery (simulated with dummy checkpoint)
2. **Target Domain**: Skin lesion classification (synthetic data)
3. **Architecture**: ResNet18 backbone with custom classification head
4. **Training**: Fine-tuning with configurable backbone freezing
5. **Evaluation**: Binary classification metrics (accuracy, precision, recall, F1, ROC-AUC)

## CLI Commands

```bash
# Individual commands
python -m src.cli prepare          # Generate synthetic data
python -m src.cli make_checkpoint  # Create dummy plant checkpoint
python -m src.cli train            # Train the model
python -m src.cli eval             # Evaluate on test set
python -m src.cli explain          # Generate Grad-CAM visualizations
python -m src.cli all              # Run complete pipeline

# With custom parameters
python -m src.cli train --epochs 10 --batch-size 16 --freeze-backbone
```

## Adapting to Real Clinical Data

**WARNING**: The following instructions are for research purposes only. Real clinical deployment requires extensive validation.

### Data Preparation
1. **Patient-Level Splitting**: Ensure no patient appears in multiple splits
2. **Class Balance**: Handle imbalanced datasets appropriately
3. **Data Quality**: Validate image quality and annotations
4. **Privacy**: Ensure proper consent and data governance

### Required Clinical Validation
- **IRB Approval**: Institutional Review Board approval required
- **Performance Targets**: Define minimum acceptable performance metrics
- **Human-in-the-Loop**: Never use as sole basis for diagnosis
- **Calibration**: Ensure model confidence aligns with accuracy
- **Adversarial Testing**: Test robustness against edge cases

### Regulatory Compliance Checklist
- [ ] FDA approval (if applicable)
- [ ] CE marking (if applicable)
- [ ] HIPAA compliance (if applicable)
- [ ] Data protection regulations
- [ ] Clinical trial registration
- [ ] Adverse event reporting system

## Known Limitations

1. **Domain Gap**: Plant cell and skin lesion images have fundamentally different characteristics
2. **Synthetic Data**: Generated images may not capture real lesion complexity
3. **Limited Dataset**: Small synthetic dataset for demonstration
4. **No Clinical Validation**: Performance on synthetic data doesn't predict real-world performance

## Recommended Next Steps

1. **Real Dataset**: Use ISIC dataset with proper patient-level splitting
2. **Domain Adaptation**: Implement adversarial domain adaptation techniques
3. **Ensemble Methods**: Combine multiple models for improved robustness
4. **Uncertainty Estimation**: Add uncertainty quantification methods
5. **Cross-Validation**: Implement k-fold cross-validation
6. **Calibration**: Ensure model confidence is well-calibrated

## Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_data.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{plant2skin_transfer,
  title={Plant2Skin Transfer Learning Pipeline},
  author={Research Team},
  year={2024},
  note={Research prototype - not for clinical use}
}
```

## Contributing

This is a research prototype. For research collaborations, please contact the maintainers.

## Support

For questions about this research prototype, please open an issue on GitHub. Remember that this is not intended for clinical use.
