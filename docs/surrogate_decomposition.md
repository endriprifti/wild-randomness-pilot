<!-- Copied unchanged from step1_decomposition_2026-11-27.tar.gz (procedure.md); the code is surrogate_decomposition.py. -->

# Step 1: splitting the surrogate baseline of the gradient series (procedure, fixed before any value is computed)

Written 2026-11-27 before the computation. Brief 9, decision 6, fixed the procedure for P6 and P6b; the same
procedure is applied to P4's two arms.

Inputs: hgrad series (steps x 8 directions) of P6 (12 runs), P6b (40), P4 transformer (95), P4 ResNet-18 (95); the
warm-up steps each H_link unit used are cut (P6/P6b 81, transformer 91, ResNet-18 234). Main scales 8 to n/10 (hurst.log_scales(n, 8)),
q = 0.5 and 4, the repository's hurst.delta_h; every statistic is the mean over the 8 directions, as in H_link's
main rows. Per run: the observed width; the mean and 95% quantile of 99 shuffled surrogates (keep the marginal only),
99 phase-randomized surrogates (keep the spectrum only, Gaussian marginal) and 99 IAAFT surrogates (keep both;
recomputed, must agree with the file's IAAFT mean within Monte Carlo error); the observed width must reproduce the
file's delta_h; the excess kurtosis of the marginal (mean over directions). Random numbers from
units.rng_for(units.seed_for(config.SEED, "step1_decomposition", key)).

P6 and P6b (separately, never pooled): P6's model (seed blocks, Noah and Joseph coded +-1/2, their product), OLS,
Noah effect with its 95% interval on each quantity; no correction; outside the family. Reading (brief 9, decision 6):
the baseline effect is assigned to the marginal if the shuffled surrogates' width shows a Noah effect of the same
sign as the IAAFT mean's, with its interval excluding 0, and the phase-randomized surrogates' does not; to the
spectrum in the reverse case; otherwise both effects are reported as found.

P4 (each arm separately): per clipping arm, the medians over seeds of the observed width and of the three surrogate
means, and the number of runs whose observed width exceeds the 95% quantile of each surrogate kind. Reading for
ResNet-18: its clipping bursts are width beyond the marginal and beyond the spectrum taken one at a time if, over the
fixed arms with targets of 1 to 50% (plain and matched, 30 runs), the observed width exceeds the shuffled and the
phase-randomized 95% quantiles in at least as many runs as it exceeds the IAAFT one (H_link's width part); otherwise
the brief reports which surrogate kind keeps the width. The transformer is described the same way. Exploratory.

Amendment, 2026-11-27 about 12:10 UTC, after one run (P6/14m_light_H0.5_s1) had been computed and before any other:
the IAAFT surrogates take about 93% of the time (0.16 s each, 792 per run; the whole job would take about 6 hours on
the two cores here). The IAAFT mean and 95% quantile that H_link's width part used are already in the result files
(hlink_grad.json for P6 and P6b, p4_grad.csv for P4). So the IAAFT values are taken from the files for every run, and
the recomputation, which was only a check, is done on 9 runs: the one already computed, the four corners of P6b's
seed 4, and seed 0's unclipped and fixed 10% runs of each P4 arm. Nothing else changes. In the run computed first, the
observed width reproduces the file's delta_h exactly and the recomputed IAAFT mean is 0.07265 against the file's
0.07267.
