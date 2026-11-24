# Methodology

## Study design and data source

This retrospective prediction study used the training data from the
PhysioNet/Computing in Cardiology Challenge 2019, which comprises hourly
intensive care unit (ICU) records from two contributing datasets, training
sets A and B. The source distribution contains one pipe-delimited file per
patient and records 34 physiological and laboratory variables, five
demographic or admission variables, ICU length of stay (ICULOS), and the
challenge sepsis label. The project audit reports 40,336 patients (20,336
from set A and 20,000 from set B) and 1,552,210 hourly observations. The
repository's development audit states that the challenge-schema files were
obtained from a public mirror and checked against the official challenge
schema; the raw data are not included in this repository. Dataset access and
use remain subject to the PhysioNet data-use agreement.

The analytic unit for partitioning was the patient. Patients were eligible
for window generation if their ICU stay contained at least 24 hourly
observations, allowing a complete 12-hour input history and at least one
complete 12-hour prediction horizon. In the repository's cohort summary,
30,661 patients met this minimum-duration criterion, of whom 1,844 were
classified as septic. The final held-out test partition contained 4,600
patients and 108,207 prediction windows. These counts describe the
repository's recorded run; they are not an independent external validation
cohort.

## Outcome definition and prediction windows

Sepsis onset was operationalized as the first ICU hour at which the
challenge-provided `SepsisLabel` equalled 1. At each eligible prediction
hour \(t\), two binary outcomes were constructed:

\[
Y_H(t) = \mathbb{1}\{t < t_{\mathrm{onset}} \leq t+H\},
\qquad H\in\{6,12\}\ \text{hours}.
\]

Thus, the outcomes indicate onset within the next 6 or 12 hours,
respectively, and the prediction time itself is not included in the future
interval. Only hours strictly before onset were considered at risk;
post-onset hours were excluded rather than treated as prediction examples.
Windows were also excluded when the full 12-hour input history was
unavailable or when fewer than 12 hours of follow-up remained before the end
of the observed stay. Such right-censored windows were not assigned a
negative label. Applying the complete 12-hour follow-up requirement to both
outcomes ensured that the 6- and 12-hour tasks used the same eligible
prediction times.

For each retained prediction time, the model input was the 12-hour sequence
from hours \(t-11\) through \(t\), inclusive. A prediction therefore used
only observations available at or before its prediction hour.

## Data processing and predictor construction

Patient records were sorted by ICULOS, and duplicate patient-hour records
were removed by retaining the first record for each patient and ICULOS.
Predictors comprised 34 clinical channels: heart rate, oxygen saturation,
temperature, systolic, mean and diastolic blood pressure, respiratory rate,
end-tidal carbon dioxide, base excess, bicarbonate, inspired oxygen
fraction, pH, arterial carbon dioxide, arterial oxygen saturation, AST,
BUN, alkaline phosphatase, calcium, chloride, creatinine, direct bilirubin,
glucose, lactate, magnesium, phosphate, potassium, total bilirubin,
troponin I, hematocrit, hemoglobin, PTT, white blood cell count,
fibrinogen, and platelets. The five demographic or admission variables
(age, gender, unit indicators, and hospital admission time) were not included
in the model input.

Missingness was represented explicitly. For each clinical channel, a
binary indicator was set to 1 when the raw value was absent and 0 otherwise.
Values were imputed by last observation carried forward within a patient
stay; any leading gaps before a channel's first observation were filled
with that channel's median calculated from training patients. Imputed
clinical values were standardized using channel-specific means and standard
deviations calculated from the training patients only. ICULOS was likewise
standardized using training-patient statistics. The resulting primary input
at each hour had 69 channels: 34 standardized clinical values, 34 binary
missingness indicators, and standardized ICULOS.

A separate 34-channel recency tensor represented elapsed time since the
most recent observation of each clinical variable. Elapsed times were
capped at 48 hours and divided by 48. For channels not yet observed, the
elapsed time was set to the cap. This tensor, together with the binary
missingness indicators, was provided to the missingness-gating branch.

## Patient-level data partition

Window-eligible patients were split into training, validation, and test
partitions in a 70:15:15 ratio. The split was stratified by patient-level
sepsis status and generated with a fixed random seed (42). Because all
windows from a patient remained in a single partition, windows from the
same patient could not occur across training, validation, and test sets.
Preprocessing statistics were estimated from training patients and then
applied unchanged to the validation and test partitions. Validation data
were used for early stopping, model selection, calibration, and threshold
selection. The test partition was reserved for final evaluation.

## Model architecture and comparator experiments

The temporal encoder was a causal temporal convolutional network (TCN)
operating on the 12-by-69 input. It comprised three convolutional blocks
with 64 channels each, kernel size 3, and dilation factors 1, 2, and 4.
Each block used batch normalization, ReLU activation, and dropout; padding
was removed from the future side so that a representation at a time point
could not depend on later inputs. Mean and maximum pooling over the temporal
dimension were concatenated to form a 128-dimensional trajectory
representation.

The informative-missingness branch flattened the 12-by-34 missingness mask
and 12-by-34 recency tensor, then mapped them through a two-layer
128-unit multilayer perceptron. It produced a missingness representation
and a sigmoid gate. The gate interpolated elementwise between the TCN
trajectory representation and the missingness representation; the fused
representation was layer-normalized. Two separate linear prediction heads
then produced logits for the 6-hour and 12-hour outcomes.

Four TCN variants were compared: the TCN alone; TCN with contrastive
pretraining; TCN with missingness gating; and TCN with both contrastive
pretraining and gating. Contrastive pretraining used a TS2Vec-inspired
instance-level objective. Two augmented views were formed using random
contiguous temporal crops of at least six hours and channel dropout on
clinical-value channels. The normalized embeddings of paired views were
matched using a symmetric InfoNCE loss (temperature 0.2); other windows in
the minibatch served as negatives. Outcome labels were not part of this
contrastive objective. For the full variant, this objective was applied to
the fused trajectory and missingness representation. The contrastive
variants were pretrained for eight epochs on the training partition before
supervised fine-tuning.

For context, logistic regression, random forest, and XGBoost classifiers
were trained separately for each horizon using the 12-by-69 windows
flattened to vectors. These tabular comparators used the same training
partition and negative-sampling strategy as the TCN experiments.
The logistic regression comparator used balanced class weights and
regularization parameter \(C=1\). The random forest used 300 trees, maximum
depth 12, and balanced-subsample class weights. XGBoost used 400 trees,
maximum depth 6, learning rate 0.1, histogram-based tree construction, and
a positive-class weight of 3. Comparator performance was summarized by
AUROC and average precision.

## Model training and selection

To reduce the computational burden and class imbalance during training,
all training windows positive for either horizon were retained, along with
a random sample of negative windows at a 3:1 negative-to-positive ratio.
This subsampling was applied to the training partition only; validation and
test windows were not subsampled. Each TCN variant was trained with the sum
of two binary cross-entropy losses with logits, one per outcome. Each loss
used a positive-class weight computed from its corresponding training
subsample. Optimization used Adam with learning rate \(5\times10^{-4}\),
weight decay \(10^{-3}\), batch size 512, and dropout 0.3. Training ran for
up to 15 supervised epochs, with early stopping after five epochs without
improvement in the mean validation average precision across the two
horizons. The best-validation checkpoint was retained. Random seeds were
fixed at 42 for model training and the training-window subsample.

The final registered model was selected from the four TCN variants by
highest mean validation average precision across the 6- and 12-hour
outcomes. In the recorded experiment this criterion selected the TCN with
missingness gating and **without** contrastive pretraining (`tcn_gate`).
Contrastive learning was therefore an evaluated experimental component, but
was not part of the selected registered model.

## Calibration, thresholds, and performance evaluation

For each model variant and prediction horizon, Platt scaling was fitted on
validation logits and validation labels using logistic regression. The
fitted transformation was then applied to test logits without refitting.
The primary binary-classification threshold was selected on validation
predictions by maximizing Youden's \(J\) statistic (sensitivity + specificity
- 1). A second, alert-oriented threshold was selected on validation data as
the smallest threshold attaining at least 80% sensitivity. Neither
calibration parameters nor thresholds were optimized on the test set.

Discrimination was summarized using the area under the receiver-operating
characteristic curve (AUROC) and average precision (reported as AUPRC).
At the validation-selected Youden threshold, the evaluation also calculated
sensitivity, specificity, precision, recall, and F1 score. Probability
quality was summarized by the Brier score. Confusion matrices were reported
for the selected model. The recorded evaluation additionally calculated
decision-curve net benefit at the Youden threshold and, among septic test
patients with a qualifying alert, the interval from the first alert above
the validation-selected 80%-sensitivity threshold to recorded onset.

Uncertainty intervals for test AUROC and AUPRC were estimated using 200
patient-level bootstrap replicates with replacement. All windows belonging
to a sampled patient were included together, preserving within-patient
clustering. The test set was evaluated after the model, calibration, and
threshold-selection procedures had been fixed.

## Reproducibility and interpretation

The repository versions feature definitions, preprocessing parameters,
model configuration, calibration coefficients, thresholds, and evaluation
outputs. The raw challenge data and trained checkpoint files are not
distributed in the repository; reproducing the full analysis requires
authorized access to the data and the corresponding generated artifacts.
The evaluation is a patient-random holdout within the combined challenge
training sets, not a held-out-hospital or external validation. Since
prediction windows overlap and the primary reported metrics are
window-level, they should not be interpreted as patient-level diagnostic
accuracy or as evidence of clinical benefit. The project is a research
prototype and not a diagnostic or autonomous treatment system. The
repository does not document an institutional review board determination;
any manuscript submission should report the applicable ethics review or
exemption and data-use approvals.

## References

1. Reyna MA, Josef C, Jeter R, et al. Early Prediction of Sepsis from
   Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019
   (version 1.0.0). *PhysioNet*. 2020.
   https://doi.org/10.13026/vc4e-7w65.
2. Bai S, Kolter JZ, Koltun V. An Empirical Evaluation of Generic
   Convolutional and Recurrent Networks for Sequence Modeling. 2018.
   arXiv:1803.01271.
3. Yue Z, Wang Y, Duan J, et al. TS2Vec: Towards Universal Representation
   of Time Series. *Proceedings of the AAAI Conference on Artificial
   Intelligence*. 2022;36(8):8980-8987.
   https://doi.org/10.1609/aaai.v36i8.20881.
4. Che Z, Purushotham S, Cho K, Sontag D, Liu Y. Recurrent Neural Networks
   for Multivariate Time Series with Missing Values. *Scientific Reports*.
   2018;8:6085. https://doi.org/10.1038/s41598-018-24271-9.
