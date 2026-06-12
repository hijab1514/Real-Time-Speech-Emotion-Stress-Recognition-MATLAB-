function res = predictEmotion(feat, model)
%PREDICTEMOTION  Module 2 (AI Prediction Engine) - inference.
%   res = ai.predictEmotion(feat) loads models/emotionModel.mat and classifies
%   the feature sequence in feat (from sigproc.extractFeatures).
%   res = ai.predictEmotion(feat, model) uses an already-loaded model struct
%   (avoids re-loading the .mat on every GUI call).
%
%   Returns struct:
%     res.label       categorical predicted class
%     res.confidence  scalar top-1 probability
%     res.scores      [1 x numClasses] probabilities
%     res.classes     class names (cellstr)
%     res.seqNorm     standardised sequence (reused by the XAI module)

arguments
    feat  (1,1) struct
    model = []
end

if isempty(model)
    here = fileparts(fileparts(mfilename('fullpath')));
    mf = fullfile(here,'models','emotionModel.mat');
    assert(isfile(mf), ...
        "No trained model found. Run ai.trainEmotionModel(datasetDir) first.");
    model = load(mf);
end

seq = (feat.seq - model.mu) ./ model.sigma;     % apply training stats
[label, scores] = classify(model.net, {seq}, SequencePaddingDirection="right");

res = struct();
res.label      = label;
res.scores     = scores;
res.classes    = string(model.classes(:)');
[res.confidence, ~] = max(scores);
res.seqNorm    = seq;
res.model      = model;     % passthrough for chained XAI call
end
