function model = trainEmotionModel(datasetDir, opts)
%TRAINEMOTIONMODEL  Module 2 (AI Prediction Engine) - training pipeline.
%   model = ai.trainEmotionModel(datasetDir) trains the hybrid CNN-LSTM on
%   ORIGINAL RAVDESS. datasetDir is the RAVDESS root, e.g.
%       datasets/audio_speech_actors_01-24
%   (the folder containing Actor_01 ... Actor_24). Files are NOT reorganised;
%   labels are parsed from RAVDESS filenames by ai.ravdessDatastore.
%
%   Options (name-value):
%     classes   (["Neutral" "Happy" "Sad" "Angry" "Fear"])  classes to keep
%     valSplit  (0.2)
%     maxEpochs (40)
%     miniBatch (32)
%     seed      (1)
%
%   Saves models/emotionModel.mat = struct(net, mu, sigma, classes,
%   valAccuracy) so gui.SERApp loads it automatically.
%
%   Requires: Audio Toolbox + Deep Learning Toolbox. GPU used if available.
%
%   NOTE (reviewed): standardisation statistics (mu/sigma) are computed on
%   the TRAINING split only and then applied to validation, avoiding the
%   data leak of fitting them on all data before the split.

arguments
    datasetDir (1,1) string
    opts.classes   (1,:) string = ["Neutral" "Happy" "Sad" "Angry" "Fear"]
    opts.valSplit  (1,1) double = 0.2
    opts.maxEpochs (1,1) double = 40
    opts.miniBatch (1,1) double = 32
    opts.seed      (1,1) double = 1
end

rng(opts.seed);

% --- 1. build labelled datastore from RAVDESS filenames -----------------
[ads, classes] = ai.ravdessDatastore(datasetDir, opts.classes);

% --- 2. feature extraction (Module 1) -----------------------------------
n = numel(ads.Files);
X = cell(n,1);
Y = ads.Labels;                 % aligned with read() order (Files order)
reset(ads);
fprintf("Extracting features");
for i = 1:n
    [x, rinfo] = read(ads);
    fs = rinfo.SampleRate;      % RAVDESS = 48 kHz; preprocess resamples
    pp = sigproc.preprocessAudio(x, fs);
    f  = sigproc.extractFeatures(pp);
    X{i} = f.seq;
    if mod(i,50)==0, fprintf("."); end
end
fprintf(" done.\n");

% drop clips too short to form a sequence
ok = cellfun(@(c) ~isempty(c) && size(c,2) >= 4, X);
X = X(ok); Y = removecats(Y(ok));
classes = categories(Y);
fprintf("Usable clips: %d, classes: %s\n", numel(X), strjoin(string(classes)', ", "));

% --- 3. stratified train/validation split -------------------------------
cv  = cvpartition(Y, HoldOut=opts.valSplit);
Xtr = X(training(cv)); Ytr = Y(training(cv));
Xva = X(test(cv));     Yva = Y(test(cv));

% --- 4. standardise per feature, fit on TRAIN ONLY ----------------------
allTr = cat(2, Xtr{:});
mu    = mean(allTr, 2);
sigma = std(allTr, 0, 2) + eps;
Xtr = cellfun(@(c) (c - mu) ./ sigma, Xtr, UniformOutput=false);
Xva = cellfun(@(c) (c - mu) ./ sigma, Xva, UniformOutput=false);

% --- 5. build + train (Module 2) ----------------------------------------
numFeatures = size(mu, 1);
minLen = min(cellfun(@(c) size(c,2), Xtr));        % shortest training sequence
layers = ai.buildCNNLSTM(numFeatures, numel(classes), minLength=minLen);

trainOpts = trainingOptions("adam", ...
    MaxEpochs           = opts.maxEpochs, ...
    MiniBatchSize       = opts.miniBatch, ...
    InitialLearnRate    = 1e-3, ...
    LearnRateSchedule   = "piecewise", ...
    LearnRateDropFactor = 0.5, ...
    LearnRateDropPeriod = 15, ...
    GradientThreshold   = 1, ...
    Shuffle             = "every-epoch", ...
    ValidationData      = {Xva, Yva}, ...
    ValidationFrequency = max(1, floor(numel(Xtr)/opts.miniBatch)), ...
    SequencePaddingDirection = "right", ...
    Plots               = "training-progress", ...
    Verbose             = true);

net = trainNetwork(Xtr, Ytr, layers, trainOpts);

% --- 6. evaluate --------------------------------------------------------
Ypred = classify(net, Xva, SequencePaddingDirection="right");
valAccuracy = mean(Ypred == Yva);
fprintf("Validation accuracy: %.1f%%\n", 100*valAccuracy);
figure; confusionchart(Yva, Ypred); title("Validation Confusion Matrix");

% --- 7. save (consumed by ai.predictEmotion / gui.SERApp) ---------------
model = struct('net', net, 'mu', mu, 'sigma', sigma, ...
               'classes', {classes}, 'valAccuracy', valAccuracy);
here = fileparts(fileparts(mfilename('fullpath')));   % project root
if ~isfolder(fullfile(here,'models')), mkdir(fullfile(here,'models')); end
save(fullfile(here,'models','emotionModel.mat'), '-struct', 'model');
fprintf("Saved models/emotionModel.mat\n");
end
