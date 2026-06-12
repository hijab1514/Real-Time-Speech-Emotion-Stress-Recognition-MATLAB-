function [x, fs] = recordMic(seconds, fs)
%RECORDMIC  Capture audio from the default microphone.
%   [x, fs] = utils.recordMic(seconds, fs) records 'seconds' of mono audio.
%   Defaults: seconds=4, fs=16000. Requires a microphone + Audio Toolbox.
arguments
    seconds (1,1) double = 4
    fs      (1,1) double = 16000
end
rec = audiorecorder(fs, 16, 1);
fprintf("Recording %.0f s ...\n", seconds);
recordblocking(rec, seconds);
x = getaudiodata(rec);
fprintf("Done.\n");
end
