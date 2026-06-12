function xai = explainPrediction(feat, pred, stress)
%EXPLAINPREDICTION  Module 2 (AI Prediction Engine) - Explainable AI.
%   xai = ai.explainPrediction(feat, pred, stress) explains WHY the CNN-LSTM
%   chose pred.label, using two complementary, defensible techniques:
%
%   1) Occlusion (perturbation) importance — model-faithful.
%      Each feature GROUP (MFCC, deltas, pitch, energy, ZCR, spectral) is
%      replaced by its mean and the drop in the predicted-class probability
%      is recorded. A large drop => that group mattered to THIS prediction.
%      This is a sound XAI method for sequence models where pixel-style
%      Grad-CAM does not apply cleanly.
%
%   2) Acoustic-profile reasoning — human-readable.
%      The global features (and the stress components) are compared to the
%      expected profile of the predicted emotion to produce plain-language
%      reasons, e.g. "high energy + fast speaking rate + high pitch variation".
%
%   Returns:
%     xai.importance   table: group, probDrop (sorted desc)
%     xai.reasons      string array of human-readable reasons
%     xai.text         single formatted explanation string

arguments
    feat   (1,1) struct
    pred   (1,1) struct
    stress (1,1) struct
end

net     = pred.model.net;
classes = pred.classes;
seq     = pred.seqNorm;                          % already standardised
clsIdx  = find(classes == string(pred.label), 1);

% baseline probability of the predicted class
[~, base] = classify(net, {seq}, SequencePaddingDirection="right");
baseP = base(clsIdx);

% --- 1. occlusion importance per feature group --------------------------
gkeys = keys(feat.groups);
gname = strings(0); pdrop = [];
for k = 1:numel(gkeys)
    idx = feat.groups(gkeys{k});
    if isempty(idx), continue; end
    occ = seq;
    occ(idx,:) = 0;                              % mean of standardised = 0
    [~, sc] = classify(net, {occ}, SequencePaddingDirection="right");
    gname(end+1,1) = string(gkeys{k});           %#ok<AGROW>
    pdrop(end+1,1) = baseP - sc(clsIdx);         %#ok<AGROW>
end
[pdrop, ord] = sort(pdrop, 'descend');
importance = table(gname(ord), pdrop, ...
    VariableNames=["FeatureGroup","ProbDrop"]);

% --- 2. acoustic-profile reasoning --------------------------------------
s = feat.summary;
reasons = strings(0);
if s.energyMean   > 0,   reasons(end+1) = local_lvl("energy", s.rmsMean, [0.05 0.2]); end
reasons(end+1) = local_lvl("speaking rate", s.speakingRate, [2.5 4.0]);
reasons(end+1) = local_lvl("pitch variation", s.pitchStd, [25 45]);
reasons(end+1) = local_lvl("spectral activity", s.fluxMean, [0.15 0.4]);
reasons(end+1) = sprintf("stress index = %.2f (%s)", stress.score, stress.level);
reasons(reasons == "") = [];

% --- 3. formatted text --------------------------------------------------
top = importance.FeatureGroup(1:min(3,height(importance)));
txt = string(sprintf("Predicted emotion: %s (%.0f%% confidence)\n", ...
    string(pred.label), 100*pred.confidence));
txt = txt + sprintf("Most influential feature groups: %s\n", strjoin(top, ", "));
txt = txt + "Acoustic profile:" + newline;
txt = txt + "  - " + strjoin(reasons, newline + "  - ");

xai = struct('importance', importance, 'reasons', reasons, 'text', txt);
end

% ========================================================================
function str = local_lvl(label, v, thr)
if     v >= thr(2), q = "high";
elseif v >= thr(1), q = "moderate";
else,               q = "low";
end
str = sprintf("%s %s (%.3g)", q, label, v);
end
