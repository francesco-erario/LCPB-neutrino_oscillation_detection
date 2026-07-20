# Measurement of the theta_13 mixing angle with neural networks and XGBoost

University project in neutrino physics. The goal is to reconstruct the theta_13 mixing angle from simulated data of a reactor experiment with a Near/Far configuration (Daya Bay style), applying machine learning techniques to classify Inverse Beta Decay (IBD) events against accidental background.

## Physics background

The experiment relies on two detectors placed at different distances from the nuclear reactor (Near ~500 m, Far ~1648 m) to measure the disappearance of electron antineutrinos due to flavor oscillation driven by theta_13. The Near detector measures the spectrum before the oscillation becomes significant, while the Far detector observes the event deficit at the point of maximum oscillation. Comparing the two spectra, corrected for the geometric dilution of the flux (1/L^2), allows the survival probability to be isolated and theta_13 to be estimated without needing precise knowledge of the absolute flux emitted by the reactor.

The experimental signature of an IBD event is a space-time coincidence between a prompt signal (positron) and a delayed signal (neutron capture). The background consists of accidental coincidences between uncorrelated events.

## Repository structure

- `dataset_setup.ipynb`: import of the raw datasets, separation between neutrino and radioactive events, construction of the basic physical and geometric features.
- `NeuralNetwork/`: feed-forward neural network model (PyTorch) for signal/background classification, with hyperparameter optimization (Optuna), training, inference, and results visualization.
- `XGBoost/`: alternative model based on XGBoost, using the same dataset and the same optimization and evaluation pipeline as the neural network.
- `neutrino_presentation.pptx`: summary presentation of the project.
- `PhysRevLett.108.171803.pdf`: reference paper (Daya Bay).

## Analysis pipeline

1. **Pair building**: identification of candidate prompt-delayed coincidences from the raw events (`build_pairs.py`, `create_datasets.py`).
2. **ML classification**: training of a neural network and an XGBoost model to distinguish IBD events from accidental background, with a cost function combining classification accuracy (AUC) and preservation of the Ep energy spectrum shape (Wasserstein distance, low-energy chi-square).
3. **Background estimation**: calculation of the accidental background via the temporal shifting technique, used both to calibrate the signal probability output of the models and to estimate false positives.
4. **Data-driven threshold**: determination of a classification threshold that makes the models' efficiency symmetric between the Near and Far detectors, correcting the bias introduced by the different signal-to-noise ratio at the two sites.
5. **Theta_13 estimation**: calculation of the oscillation probability from the comparison of the Near/Far spectra, using three independent methods: bin-by-bin estimation with error propagation, weighted mean, and Poissonian fit of the expected Far spectrum starting from the Near spectrum.
6. **Purity and calibration**: evaluation of the purity of the selected sample both through the classifier's calibrated probabilities and through the independent background estimate.

## Models

Both models (neural network in `NeuralNetwork/`, XGBoost in `XGBoost/`) are trained on the same dataset and the same physical and geometric features, with hyperparameter optimization via Optuna and 5-fold cross-validation. The trained models and their scalers are saved in the respective folders together with the inference notebooks (`inferenceNN.ipynb`, `inferenceXGB.ipynb`) and the results analysis notebooks (`NN_analysis.ipynb`, `XGB_analysis.ipynb`).
