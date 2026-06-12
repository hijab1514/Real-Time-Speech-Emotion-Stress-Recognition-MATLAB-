function layers = buildCNNLSTM(numFeatures, numClasses, p)
%BUILDCNNLSTM  Module 2 (AI Prediction Engine) - network definition.
%   layers = ai.buildCNNLSTM(numFeatures, numClasses) returns a layer array
%   for a hybrid 1-D CNN -> BiLSTM -> FC -> softmax classifier operating on
%   per-frame feature sequences [numFeatures x time].
%
%   Architecture rationale:
%     CNN front-end  : learns local spectro-temporal patterns across frames
%                      and reduces the time axis (cheaper, denoised input
%                      to the recurrent stage).
%     BiLSTM         : models how those patterns evolve over the utterance
%                      in both directions (emotion cues are non-causal).
%     FC + softmax   : maps the final state to class probabilities.
%
%   Options (name-value):
%     numFilters (64)   base conv filters (doubled in 2nd block)
%     filterSize (5)    conv kernel length (frames)
%     lstmHidden (128)  BiLSTM hidden units
%     dropout    (0.3)
%
%   Requires: Deep Learning Toolbox R2021b+ (convolution1dLayer,
%   maxPooling1dLayer). For R2024a+ replace classificationLayer/softmax with
%   trainnet + "crossentropy" loss (see trainEmotionModel notes).

arguments
    numFeatures (1,1) double
    numClasses  (1,1) double
    p.numFilters (1,1) double = 64
    p.filterSize (1,1) double = 5
    p.lstmHidden (1,1) double = 128
    p.dropout    (1,1) double = 0.3
    p.minLength  (1,1) double = 4 
end

layers = [
    sequenceInputLayer(numFeatures, Name="input", Normalization="none" ,MinLength=p.minLength)

    convolution1dLayer(p.filterSize, p.numFilters, Padding="same", Name="conv1")
    batchNormalizationLayer(Name="bn1")
    reluLayer(Name="relu1")
    maxPooling1dLayer(2, Stride=2, Name="pool1")

    convolution1dLayer(p.filterSize, 2*p.numFilters, Padding="same", Name="conv2")
    batchNormalizationLayer(Name="bn2")
    reluLayer(Name="relu2")
    maxPooling1dLayer(2, Stride=2, Name="pool2")

    bilstmLayer(p.lstmHidden, OutputMode="last", Name="bilstm")
    dropoutLayer(p.dropout, Name="drop")

    fullyConnectedLayer(64, Name="fc1")
    reluLayer(Name="relu3")
    fullyConnectedLayer(numClasses, Name="fc2")
    softmaxLayer(Name="softmax")
    classificationLayer(Name="output")
];
end
