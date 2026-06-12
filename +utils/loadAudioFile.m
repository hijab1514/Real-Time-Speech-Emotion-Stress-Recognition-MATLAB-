function [x, fs, fname] = loadAudioFile(path)
%LOADAUDIOFILE  Read an audio file (.wav/.flac/.mp3/.ogg/.m4a).
%   [x, fs, fname] = utils.loadAudioFile(path). If 'path' is omitted a file
%   picker is shown.
arguments
    path (1,1) string = ""
end
if path == ""
    [f, d] = uigetfile({'*.wav;*.flac;*.mp3;*.ogg;*.m4a','Audio files'}, ...
                       'Select an audio file');
    if isequal(f,0), x=[]; fs=[]; fname=""; return; end
    path = fullfile(d, f);
end
[x, fs] = audioread(path);
[~, n, e] = fileparts(path);
fname = string(n) + string(e);
end
