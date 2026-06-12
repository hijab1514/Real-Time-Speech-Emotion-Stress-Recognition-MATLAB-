function tests = test_pipeline
%TEST_PIPELINE  Smoke + unit tests for Modules 1 and 2 (stress path).
%   Run from project root:  results = runtests("tests/test_pipeline.m")
%   These tests need Audio Toolbox but NOT a trained model (they exercise
%   preprocessing, feature extraction, and the rule-based stress engine on a
%   synthetic voiced signal).
tests = functiontests(localfunctions);
end

function setupOnce(tc)
% put project root on path
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
tc.TestData.root = root;
% synthetic 2 s "voiced" signal: 150 Hz carrier + harmonics + noise
fs = 16000; t = (0:1/fs:2)';
x = 0.6*sin(2*pi*150*t) + 0.3*sin(2*pi*300*t) + 0.05*randn(size(t));
tc.TestData.x = x; tc.TestData.fs = fs;
end

function testPreprocessShape(tc)
pp = sigproc.preprocessAudio(tc.TestData.x, tc.TestData.fs);
verifyEqual(tc, pp.fs, 16000);
verifyGreaterThan(tc, numel(pp.signal), 0);
verifyLessThanOrEqual(tc, max(abs(pp.signal)), 1+1e-6);   % normalised
end

function testFeatureSequence(tc)
pp   = sigproc.preprocessAudio(tc.TestData.x, tc.TestData.fs);
feat = sigproc.extractFeatures(pp);
verifySize(tc, feat.seq, [numel(feat.names) size(feat.seq,2)]);
verifyTrue(tc, all(isfinite(feat.seq(:))));
verifyTrue(tc, isfield(feat.summary,'pitchStd'));
end

function testStressEngineRange(tc)
pp   = sigproc.preprocessAudio(tc.TestData.x, tc.TestData.fs);
feat = sigproc.extractFeatures(pp);
s    = ai.stressEngine(feat);
verifyGreaterThanOrEqual(tc, s.score, 0);
verifyLessThanOrEqual(tc, s.score, 1);
verifyTrue(tc, ismember(s.level, ["Low","Moderate","High"]));
end
