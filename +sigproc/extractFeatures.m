function feat = extractFeatures(pp)
%EXTRACTFEATURES  Module 1 (Signal Processing Engine) feature engine.
%   feat = sigproc.extractFeatures(pp) takes the struct returned by
%   sigproc.preprocessAudio and returns:
%       feat.seq        [numFeatures x numFrames]  -> input to the CNN-LSTM
%       feat.names      {1 x numFeatures}          -> feature/channel names
%       feat.groups     containers.Map name->idx   -> for XAI grouping
%       feat.summary    struct of scalar/global features (stress + XAI)
%
%   Sequence features (per frame) come from audioFeatureExtractor:
%     MFCC(13) + dMFCC(13) + ddMFCC(13), spectral centroid / rolloff /
%     flux / spread / entropy, pitch, ZCR, short-time energy, harmonic ratio.
%
%   Summary features (per utterance) are used by the stress engine and the
%   natural-language explainer.
%
%   Requires: Audio Toolbox (audioFeatureExtractor, pitch).

arguments
    pp (1,1) struct
end

x  = pp.signal;
fs = pp.fs;
win = pp.window;
ov  = pp.frameLen - pp.hopLen;

% --- per-frame feature set ----------------------------------------------
afe = audioFeatureExtractor( ...
    SampleRate        = fs, ...
    Window            = win, ...
    OverlapLength     = ov, ...
    mfcc              = true, ...
    mfccDelta         = true, ...
    mfccDeltaDelta    = true, ...
    spectralCentroid  = true, ...
    spectralRolloffPoint = true, ...
    spectralFlux      = true, ...
    spectralSpread    = true, ...
    spectralEntropy   = true, ...
    pitch             = true, ...
    zerocrossrate     = true, ...
    shortTimeEnergy   = true, ...
    harmonicRatio     = true);

F = extract(afe, x);                 % [numFrames x numFeatures]
F(~isfinite(F)) = 0;

names = local_channelNames(afe);

seq = F.';                           % [numFeatures x numFrames]

% --- summary (global) features for stress + explanation -----------------
[f0, ~] = pitch(x, fs, WindowLength=pp.frameLen, OverlapLength=ov, ...
                Range=[50 400]);
f0 = f0(isfinite(f0));
energy = sum(buffer(x, pp.frameLen, ov, 'nodelay').^2, 1);

s = struct();
s.duration        = pp.duration;
s.pitchMean       = local_safe(@mean, f0);
s.pitchStd        = local_safe(@std,  f0);          % pitch instability
s.pitchRange      = local_safe(@(v) max(v)-min(v), f0);
s.energyMean      = mean(energy);
s.energyStd       = std(energy);                    % energy fluctuation
s.energyCV        = s.energyStd / max(s.energyMean, eps);
s.rmsMean         = sqrt(mean(x.^2));
s.zcrMean         = mean(F(:, contains(names,'zerocrossrate')), 'omitnan');
s.centroidMean    = mean(F(:, strcmp(names,'spectralCentroid')), 'omitnan');
s.fluxMean        = mean(F(:, strcmp(names,'spectralFlux')), 'omitnan');
s.speakingRate    = local_speakingRate(x, fs, pp);  % voiced segments / sec

% group index map for XAI (channel ranges by feature family)
groups = containers.Map('KeyType','char','ValueType','any');
groups('MFCC')        = find(startsWith(names,'mfcc_'));
groups('Delta-MFCC')  = find(startsWith(names,'mfccDelta_'));
groups('DD-MFCC')     = find(startsWith(names,'mfccDeltaDelta_'));
groups('Pitch')       = find(strcmp(names,'pitch'));
groups('Energy')      = find(strcmp(names,'shortTimeEnergy'));
groups('ZCR')         = find(strcmp(names,'zerocrossrate'));
groups('Spectral')    = find(ismember(names, ...
    {'spectralCentroid','spectralRolloffPoint','spectralFlux', ...
     'spectralSpread','spectralEntropy','harmonicRatio'}));

feat = struct('seq', seq, 'names', {names}, 'groups', groups, 'summary', s);
end

% ========================================================================
function names = local_channelNames(afe)
i = info(afe);
names = {};
flds = fieldnames(i);
for k = 1:numel(flds)
    n = numel(i.(flds{k}));
    if n == 1
        names{end+1} = flds{k};                     %#ok<AGROW>
    else
        for j = 1:n
            names{end+1} = sprintf('%s_%d', flds{k}, j); %#ok<AGROW>
        end
    end
end
end

function r = local_speakingRate(x, fs, pp)
% crude syllable proxy: count voiced runs via short-time energy peaks
ov  = pp.frameLen - pp.hopLen;
e   = sum(buffer(x, pp.frameLen, ov, 'nodelay').^2, 1);
e   = e / max(e);
voiced = e > 0.15;
runs = sum(diff([0 voiced 0]) == 1);
r = runs / max(pp.duration, eps);
end

function v = local_safe(fn, data)
if isempty(data), v = 0; else, v = fn(data); end
end
