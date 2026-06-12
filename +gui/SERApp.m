classdef SERApp < handle
%SERAPP  Module 3 (GUI) - Speech Emotion & Stress Recognition application.
%   app = gui.SERApp;  launches the interface.
%
%   Built programmatically with uifigure/uigridlayout so the entire UI is in
%   source control (no binary .mlapp). It is functionally identical to an App
%   Designer app and you can paste this layout logic into App Designer's
%   startupFcn if your rubric requires the .mlapp file specifically.
%
%   The GUI is a THIN layer: it only acquires audio and calls the DSP and AI
%   engines. No analysis logic lives here — that is the point of the 3-module
%   separation.
%
%   Panels: Live Recording | Audio Upload | Waveform | Spectrogram | MFCC |
%           Emotion + Confidence | Stress | Explanation | History | Export.
%
%   Requires: Audio Toolbox, Deep Learning Toolbox, a trained model
%   (models/emotionModel.mat) for emotion output.

    properties
        Fig
        % controls
        RecordBtn; StopBtn; UploadBtn; ExportBtn; DurField
        % displays
        WaveAx; SpecAx; MfccAx
        EmotionLamp; EmotionLbl; ConfGauge
        StressLbl; StressGauge; ExplainArea; HistTable
        % state
        Recorder; Model; LastResult; RecTimer
    end

    methods
        function app = SERApp()
            app.loadModel();
            app.buildUI();
        end
    end

    methods (Access = private)
        % ---------------------------------------------------------------
        function loadModel(app)
            here = fileparts(fileparts(mfilename('fullpath')));
            mf = fullfile(here,'models','emotionModel.mat');
            if isfile(mf)
                app.Model = load(mf);
            else
                app.Model = [];
                warning("No trained model. Emotion output disabled until " + ...
                        "ai.trainEmotionModel runs.");
            end
        end

        % ---------------------------------------------------------------
        function buildUI(app)
            app.Fig = uifigure(Name="Speech Emotion & Stress Recognition", ...
                               Position=[100 100 1180 720], Color=[0.96 0.97 0.99]);
            g = uigridlayout(app.Fig, [3 3]);
            g.RowHeight    = {120, '1x', 190};
            g.ColumnWidth  = {'1.4x', '1.4x', '1x'};
            g.Padding = [12 12 12 12]; g.RowSpacing = 10; g.ColumnSpacing = 10;

            % ---- row 1: control bar (spans 3 cols) ---------------------
            ctrl = uipanel(g, Title="Controls"); ctrl.Layout.Row = 1; ctrl.Layout.Column = [1 3];
            cg = uigridlayout(ctrl,[1 6]); cg.ColumnWidth = {110,110,60,130,'1x',130};
            app.RecordBtn = uibutton(cg, Text="● Record", ...
                FontWeight="bold", BackgroundColor=[0.85 0.2 0.2], FontColor="w", ...
                ButtonPushedFcn=@(~,~) app.onRecord());
            app.StopBtn = uibutton(cg, Text="■ Stop", Enable="off", ...
                ButtonPushedFcn=@(~,~) app.onStop());
            uilabel(cg, Text="sec:");
            app.DurField = uieditfield(cg,"numeric", Value=4, Limits=[1 30]);
            app.UploadBtn = uibutton(cg, Text="⬆ Upload Audio", ...
                ButtonPushedFcn=@(~,~) app.onUpload());
            app.ExportBtn = uibutton(cg, Text="⤓ Export Result", Enable="off", ...
                ButtonPushedFcn=@(~,~) app.onExport());

            % ---- row 2 col 1: waveform + spectrogram -------------------
            vp = uipanel(g, Title="Signal"); vp.Layout.Row = 2; vp.Layout.Column = 1;
            vgr = uigridlayout(vp,[2 1]);
            app.WaveAx = uiaxes(vgr); title(app.WaveAx,"Waveform");
            app.SpecAx = uiaxes(vgr); title(app.SpecAx,"Spectrogram");

            % ---- row 2 col 2: MFCC + explanation -----------------------
            mp = uipanel(g, Title="Features & Explanation"); mp.Layout.Row = 2; mp.Layout.Column = 2;
            mgr = uigridlayout(mp,[2 1]); mgr.RowHeight = {'1x','1x'};
            app.MfccAx = uiaxes(mgr); title(app.MfccAx,"MFCC");
            app.ExplainArea = uitextarea(mgr, Editable="off", ...
                Value="Explanation will appear here.", FontName="monospaced");

            % ---- row 2 col 3: results ----------------------------------
            rp = uipanel(g, Title="Prediction"); rp.Layout.Row = 2; rp.Layout.Column = 3;
            rgr = uigridlayout(rp,[6 1]); rgr.RowHeight = {30,40,70,30,70,'1x'};
            uilabel(rgr, Text="Emotion", FontWeight="bold");
            app.EmotionLbl = uilabel(rgr, Text="—", FontSize=26, ...
                FontWeight="bold", HorizontalAlignment="center");
            app.ConfGauge = uigauge(rgr,"linear", Limits=[0 100]);
            uilabel(rgr, Text="Stress", FontWeight="bold");
            app.StressGauge = uigauge(rgr,"linear", Limits=[0 100], ...
                ScaleColors=["green","yellow","red"], ScaleColorLimits=[0 40;40 70;70 100]);
            app.StressLbl = uilabel(rgr, Text="—", FontSize=16, ...
                HorizontalAlignment="center");

            % ---- row 3: history (spans 3 cols) -------------------------
            hp = uipanel(g, Title="Prediction History"); hp.Layout.Row = 3; hp.Layout.Column = [1 3];
            hgr = uigridlayout(hp,[1 1]);
            app.HistTable = uitable(hgr, ...
                ColumnName=["Time","File","Emotion","Conf %","Stress","Stress idx"], ...
                Data=cell(0,6));
        end

        % ---------------------------------------------------------------
        function onRecord(app)
            fs = 16000;
            app.Recorder = audiorecorder(fs,16,1);
            record(app.Recorder);
            app.RecordBtn.Enable = "off"; app.StopBtn.Enable = "on";
            % auto-stop after DurField seconds (stored so we can clean it up)
            app.RecTimer = timer(StartDelay=app.DurField.Value, ...
                      ExecutionMode="singleShot", ...
                      TimerFcn=@(~,~) app.onStop());
            start(app.RecTimer);
        end

        function onStop(app)
            % stop & delete the auto-stop timer if it is still around
            if ~isempty(app.RecTimer) && isvalid(app.RecTimer)
                stop(app.RecTimer); delete(app.RecTimer); app.RecTimer = [];
            end
            if isempty(app.Recorder) || ~isrecording(app.Recorder)
                return;
            end
            stop(app.Recorder);
            x  = getaudiodata(app.Recorder);
            fs = app.Recorder.SampleRate;
            app.RecordBtn.Enable = "on"; app.StopBtn.Enable = "off";
            app.analyse(x, fs, "live_mic");
        end

        function onUpload(app)
            [x, fs, fname] = utils.loadAudioFile();
            if isempty(x), return; end
            app.analyse(x, fs, fname);
        end

        % ---------------------------------------------------------------
        function analyse(app, x, fs, fname)
            d = uiprogressdlg(app.Fig, Title="Analysing", Indeterminate="on");
            cleaner = onCleanup(@() close(d)); %#ok<NASGU>

            pp   = sigproc.preprocessAudio(x, fs);
            feat = sigproc.extractFeatures(pp);

            % --- plots (defensive: a plotting failure must not crash the app) ---
            try
                app.drawPlots(pp, feat);
            catch ME
                warning("Plotting failed: %s", ME.message);
            end

            % stress always available
            stress = ai.stressEngine(feat);
            app.StressGauge.Value = 100*stress.score;
            app.StressLbl.Text = stress.level;

            % emotion only if a model is loaded
            if isempty(app.Model)
                app.EmotionLbl.Text = "no model";
                app.ConfGauge.Value = 0;
                app.ExplainArea.Value = ...
                    "Train a model (ai.trainEmotionModel) to enable emotion + XAI.";
                emoStr = "N/A"; conf = 0;
            else
                try
                    pred = ai.predictEmotion(feat, app.Model);
                    xai  = ai.explainPrediction(feat, pred, stress);
                    emoStr = string(pred.label); conf = pred.confidence;
                    app.EmotionLbl.Text = emoStr;
                    app.ConfGauge.Value = 100*conf;
                    app.ExplainArea.Value = splitlines(string(xai.text));
                    app.LastResult = struct('file',fname,'pred',pred, ...
                        'stress',stress,'xai',xai);
                catch ME
                    app.EmotionLbl.Text = "error";
                    app.ConfGauge.Value = 0;
                    app.ExplainArea.Value = "Prediction failed: " + string(ME.message);
                    emoStr = "ERROR"; conf = 0;
                end
            end

            % history row + enable export
            newRow = {char(datetime("now","Format","HH:mm:ss")), char(fname), ...
                      char(emoStr), round(100*conf), char(stress.level), ...
                      round(stress.score,2)};
            app.HistTable.Data = [app.HistTable.Data; newRow];
            app.ExportBtn.Enable = "on";
        end

        % ---------------------------------------------------------------
        function drawPlots(app, pp, feat)
        % Renders waveform, spectrogram and MFCC into the UIAxes.
        % NOTE: spectrogram() has no spectrogram(ax,...) syntax. Passing a
        % UIAxes as the first argument makes MATLAB treat the axes as the
        % signal -> "Expected X to be ... double". The correct pattern for
        % App Designer / uifigure is to compute the STFT with OUTPUT
        % arguments and draw it with imagesc into the target axes.

            % --- waveform ---
            cla(app.WaveAx);
            t = (0:numel(pp.signal)-1) / pp.fs;
            plot(app.WaveAx, t, pp.signal);
            title(app.WaveAx, "Waveform");
            xlabel(app.WaveAx, "s"); ylabel(app.WaveAx, "amp");
            axis(app.WaveAx, "tight"); grid(app.WaveAx, "on");

            % --- spectrogram (computed, then drawn into UIAxes) ---
            cla(app.SpecAx);
            win  = pp.window;
            nov  = min(pp.frameLen - pp.hopLen, numel(win) - 1);   % must be < window length
            nfft = max(256, 2^nextpow2(pp.frameLen));
            [S, F, T] = spectrogram(pp.signal, win, nov, nfft, pp.fs);
            P = 20*log10(abs(S) + eps);                            % power in dB
            imagesc(app.SpecAx, T, F, P);
            app.SpecAx.YDir = "normal";
            title(app.SpecAx, "Spectrogram");
            xlabel(app.SpecAx, "s"); ylabel(app.SpecAx, "Hz");
            axis(app.SpecAx, "tight");

            % --- MFCC heatmap ---
            cla(app.MfccAx);
            mfcc = feat.seq(startsWith(feat.names, 'mfcc_'), :);
            imagesc(app.MfccAx, mfcc);
            app.MfccAx.YDir = "normal";
            title(app.MfccAx, "MFCC");
            xlabel(app.MfccAx, "frame"); ylabel(app.MfccAx, "coeff");
            axis(app.MfccAx, "tight");
        end

        % ---------------------------------------------------------------
        function onExport(app)
            if isempty(app.LastResult)
                uialert(app.Fig,"Nothing to export yet.","Export"); return;
            end
            r = app.LastResult;
            rec = struct('file', r.file, 'timestamp', string(datetime("now")), ...
                'emotion', string(r.pred.label), 'confidence', r.pred.confidence, ...
                'stress', r.stress.level, 'stressScore', r.stress.score, ...
                'reasons', char(r.xai.text));
            out = utils.exportReport(rec);
            uialert(app.Fig, "Saved: " + out, "Export", Icon="success");
        end
    end
end
