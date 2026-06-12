function out = preprocessAudio(x, fs, opts)
%PREPROCESSAUDIO  Module 1 (Signal Processing Engine) front-end.
%   out = sigproc.preprocessAudio(x, fs) returns a struct with a cleaned signal
%   and the metadata needed by the feature extractor.
%
%   Pipeline:  mono -> resample -> pre-emphasis -> spectral-subtraction
%   denoise -> voice activity detection (silence removal) -> peak normalize.
%
%   Name-value options:
%     targetFs   (16000)  working sample rate
%     frameDur   (0.025)  analysis frame length [s]
%     hopDur     (0.010)  hop / stride [s]
%     preEmph    (0.97)   pre-emphasis coefficient
%     doDenoise  (true)   spectral subtraction on/off
%     doVAD      (true)   trim non-speech using detectSpeech
%
%   Requires: Audio Toolbox (detectSpeech). Falls back to energy gating if
%   detectSpeech is unavailable.

arguments
    x  (:,:) double
    fs (1,1) double
    opts.targetFs  (1,1) double = 16000
    opts.frameDur  (1,1) double = 0.025
    opts.hopDur    (1,1) double = 0.010
    opts.preEmph   (1,1) double = 0.97
    opts.doDenoise (1,1) logical = true
    opts.doVAD     (1,1) logical = true
end

% --- 1. Mono -------------------------------------------------------------
if size(x,2) > 1
    x = mean(x,2);          % downmix channels
end
x = x(:);

% --- 2. Resample to working rate ----------------------------------------
if fs ~= opts.targetFs
    x  = resample(x, opts.targetFs, fs);
    fs = opts.targetFs;
end

% --- 3. Pre-emphasis (boosts high frequencies / flattens spectrum) ------
x = filter([1 -opts.preEmph], 1, x);

% --- 4. Spectral-subtraction denoise ------------------------------------
% Estimates the noise floor from the first 250 ms (assumed non-speech)
% and subtracts it in the magnitude-STFT domain.
if opts.doDenoise && numel(x) > fs*0.3
    x = local_spectralSubtract(x, fs);
end

% --- 5. Voice Activity Detection / silence removal ----------------------
speechIdx = [];
if opts.doVAD
    try
        idx = detectSpeech(x, fs);              % Audio Toolbox
        if ~isempty(idx)
            speechIdx = idx;
            keep = false(numel(x),1);
            for k = 1:size(idx,1)
                keep(idx(k,1):idx(k,2)) = true;
            end
            if any(keep), x = x(keep); end
        end
    catch
        x = local_energyGate(x, fs);            % fallback
    end
end

% --- 6. Peak normalization ----------------------------------------------
pk = max(abs(x));
if pk > 0, x = x ./ pk; end

% --- output --------------------------------------------------------------
out = struct();
out.signal     = x;
out.fs         = fs;
out.frameLen   = round(opts.frameDur * fs);
out.hopLen     = round(opts.hopDur   * fs);
out.window     = hann(out.frameLen, 'periodic');
out.duration   = numel(x) / fs;
out.speechIdx  = speechIdx;
end

% ========================================================================
function y = local_spectralSubtract(x, fs)
win  = round(0.025*fs); hop = round(0.010*fs);
nfft = 2^nextpow2(win);
[S,~,~] = stft(x, fs, 'Window', hann(win,'periodic'), ...
               'OverlapLength', win-hop, 'FFTLength', nfft);
mag = abs(S); ph = angle(S);
noiseFrames = max(1, round(0.25*fs/hop));
noiseMag = mean(mag(:,1:min(noiseFrames,size(mag,2))), 2);
cleanMag = max(mag - 1.0*noiseMag, 0.05*mag);     % over-subtraction + floor
Sclean   = cleanMag .* exp(1i*ph);
y = istft(Sclean, fs, 'Window', hann(win,'periodic'), ...
          'OverlapLength', win-hop, 'FFTLength', nfft);
y = real(y);
end

% ========================================================================
function y = local_energyGate(x, fs)
win = round(0.02*fs);
e   = movmean(x.^2, win);
thr = 0.1 * max(e);
y   = x(e > thr);
if isempty(y), y = x; end
end
