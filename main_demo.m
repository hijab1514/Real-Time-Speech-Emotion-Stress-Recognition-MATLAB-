function main_demo(audioPath)
%MAIN_DEMO  End-to-end command-line run of the full pipeline (no GUI).
%   main_demo()           -> records 4 s from the microphone, then analyses.
%   main_demo("clip.wav") -> analyses an existing file.
%
%   Useful for unit/integration testing and for verifying the pipeline
%   before the live GUI demonstration. Run this from the project root so the
%   +sigproc/+ai/+utils packages are on the path.
%
%   Flow: acquire -> preprocess -> features -> emotion -> stress -> explain
%         -> visualise -> export.

arguments
    audioPath (1,1) string = ""
end

% --- 1. acquire ----------------------------------------------------------
if audioPath == ""
    [x, fs] = utils.recordMic(4, 16000);
    fname = "live_mic";
else
    [x, fs, fname] = utils.loadAudioFile(audioPath);
end

% --- 2-3. preprocess + features (Module 1) ------------------------------
pp   = sigproc.preprocessAudio(x, fs);
feat = sigproc.extractFeatures(pp);

% --- 4-6. predict + stress + explain (Module 2) -------------------------
pred   = ai.predictEmotion(feat);                 % needs trained model
stress = ai.stressEngine(feat);
xai    = ai.explainPrediction(feat, pred, stress);

% --- console output -----------------------------------------------------
fprintf("\n%s\n", xai.text);

% --- visualise ----------------------------------------------------------
figure(Name="SER pipeline", Color="w");
tiledlayout(3,1);
nexttile; plot((0:numel(pp.signal)-1)/pp.fs, pp.signal);
title("Waveform"); xlabel("s"); ylabel("amp"); grid on;
nexttile; spectrogram(pp.signal, pp.window, pp.frameLen-pp.hopLen, ...
                      [], pp.fs, "yaxis"); title("Spectrogram");
nexttile; imagesc(feat.seq(startsWith(feat.names,'mfcc_'),:));
axis xy; title("MFCC sequence"); xlabel("frame"); ylabel("coeff"); colorbar;

% --- export -------------------------------------------------------------
rec = struct('file', fname, 'timestamp', string(datetime("now")), ...
    'emotion', string(pred.label), 'confidence', pred.confidence, ...
    'stress', stress.level, 'stressScore', stress.score, ...
    'reasons', char(xai.text));
out = utils.exportReport(rec);
fprintf("Report written: %s\n", out);
end
