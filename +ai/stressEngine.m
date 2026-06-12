function res = stressEngine(feat, calib)
%STRESSENGINE  Module 2 (AI Prediction Engine) - stress subsystem.
%   res = ai.stressEngine(feat) estimates a stress level from the global
%   acoustic features in feat.summary (from sigproc.extractFeatures).
%
%   IMPORTANT (design note): the four standard SER datasets (RAVDESS, TESS,
%   CREMA-D, SAVEE) do NOT carry stress labels. Stress here is therefore an
%   UNSUPERVISED, rule-based index built from validated acoustic correlates
%   of vocal stress, not a trained classifier. Treat the thresholds as a
%   demonstrable heuristic that must be CALIBRATED on your own recordings
%   before any quantitative claim. This is stated plainly in the report so it
%   is defensible under questioning.
%
%   Stress correlates used (each increases with stress):
%     - pitch instability   (std of F0)
%     - pitch level          (mean F0)
%     - energy fluctuation   (coefficient of variation of frame energy)
%     - speaking rate        (voiced segments / second)
%     - spectral flux        (rate of spectral change)
%
%   Returns:
%     res.level        "Low" | "Moderate" | "High"
%     res.score        composite stress index in [0,1]
%     res.components   table: feature, value, normalised, weight, contribution
%
%   calib (optional) overrides the per-feature normalisation ranges
%   [min max] obtained from your calibration set.

arguments
    feat  (1,1) struct
    calib (1,1) struct = local_defaultCalib()
end

s = feat.summary;

% raw -> normalised [0,1] via calibration ranges
nz = @(v, r) min(max((v - r(1)) / (r(2) - r(1) + eps), 0), 1);

comp = {
%   name              raw value          range            weight
    "Pitch instability" s.pitchStd        calib.pitchStd    0.30
    "Pitch level"       s.pitchMean       calib.pitchMean   0.15
    "Energy variation"  s.energyCV        calib.energyCV    0.25
    "Speaking rate"     s.speakingRate    calib.speakRate   0.20
    "Spectral flux"     s.fluxMean        calib.flux        0.10
};

name = string(comp(:,1));
raw  = cell2mat(comp(:,2));
ranges = comp(:,3);
w    = cell2mat(comp(:,4));

normv = arrayfun(@(i) nz(raw(i), ranges{i}), (1:numel(raw))');
contrib = normv .* w;
score = sum(contrib) / sum(w);

if     score < 0.40, level = "Low";
elseif score < 0.70, level = "Moderate";
else,                level = "High";
end

components = table(name, raw, normv, w, contrib, ...
    VariableNames=["Feature","Value","Normalised","Weight","Contribution"]);

res = struct('level', level, 'score', score, 'components', components);
end

% ========================================================================
function c = local_defaultCalib()
% Placeholder ranges [min max] for normalisation. REPLACE with values
% measured on your calibration recordings (relaxed reading vs. timed stressor).
c.pitchStd   = [10  60];     % Hz
c.pitchMean  = [90  260];    % Hz
c.energyCV   = [0.2 1.2];
c.speakRate  = [1.5 5.0];    % voiced segments / s
c.flux       = [0.05 0.6];
end
