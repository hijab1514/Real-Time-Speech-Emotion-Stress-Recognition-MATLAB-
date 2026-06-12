function [ads, classes] = ravdessDatastore(datasetDir, supportedClasses)
%RAVDESSDATASTORE  Build a labelled datastore for ORIGINAL RAVDESS.
%   [ads, classes] = ai.ravdessDatastore(datasetDir) recursively scans the
%   RAVDESS tree (datasetDir/audio_speech_actors_01-24/Actor_##/*.wav) and
%   labels every file from the emotion code in its filename, so you do NOT
%   reorganise files into emotion folders.
%
%   RAVDESS filename: MM-VV-EE-II-SS-RR-AA.wav  (7 hyphen tokens)
%       token #3 (EE) = emotion code:
%         01 Neutral  02 Calm  03 Happy  04 Sad
%         05 Angry    06 Fear  07 Disgust 08 Surprised
%   (RAVDESS calls 06 "fearful"; we label it "Fear" to match the project's
%   5-class scheme.)
%
%   supportedClasses (string row) selects which classes to keep; everything
%   else is discarded. Default keeps the five required by the project:
%       ["Neutral" "Happy" "Sad" "Angry" "Fear"]
%   Pass e.g. ["Neutral" "Calm" "Happy" "Sad" "Angry" "Fear" "Disgust" ...
%   "Surprised"] to use all eight.
%
%   Returns ads (with .Labels set as a categorical) and the present classes.

arguments
    datasetDir       (1,1) string
    supportedClasses (1,:) string = ["Neutral" "Happy" "Sad" "Angry" "Fear"]
end

assert(isfolder(datasetDir), "Dataset folder not found: %s", datasetDir);

% 1. recursively index every wav under the tree (Actor_* and any subfolder)
ads = audioDatastore(datasetDir, IncludeSubfolders=true, FileExtensions=".wav");
assert(~isempty(ads.Files), "No .wav files found under %s", datasetDir);

% RAVDESS emotion-code -> name
codeKeys = {'01','02','03','04','05','06','07','08'};
codeVals = {'Neutral','Calm','Happy','Sad','Angry','Fear','Disgust','Surprised'};
codeMap  = containers.Map(codeKeys, codeVals);

% 2-3. parse label from filename token #3
n   = numel(ads.Files);
raw = strings(n,1);
for i = 1:n
    [~, name] = fileparts(ads.Files{i});      % drop path + ".wav"
    tok = split(string(name), "-");
    if numel(tok) >= 3 && isKey(codeMap, char(tok(3)))
        raw(i) = string(codeMap(char(tok(3))));
    else
        raw(i) = "Unknown";                    % not a RAVDESS-style name
    end
end

% 4. keep only supported classes
keep = ismember(raw, supportedClasses);
assert(any(keep), ...
    "No files matched RAVDESS naming / the requested classes. " + ...
    "Point datasetDir at the RAVDESS root that contains the Actor_* folders.");

ads = subset(ads, find(keep));
% fixed category order = supportedClasses; removecats drops any with 0 files
ads.Labels = removecats(categorical(raw(keep), supportedClasses));
classes = categories(ads.Labels);

% report (incl. per-class counts so you can see RAVDESS' class imbalance)
cnt = countcats(ads.Labels);
fprintf("RAVDESS: kept %d/%d files across %d classes.\n", nnz(keep), n, numel(classes));
for k = 1:numel(classes)
    fprintf("   %-10s %4d\n", classes{k}, cnt(k));
end
end
