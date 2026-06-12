function outFile = exportReport(record, outDir)
%EXPORTREPORT  Persist one prediction record to CSV (appends) + a text card.
%   outFile = utils.exportReport(record, outDir). 'record' is a struct with
%   fields: file, timestamp, emotion, confidence, stress, stressScore, reasons.
%   Appends a row to reports/history.csv and writes a per-run .txt summary.
arguments
    record (1,1) struct
    outDir (1,1) string = fullfile(fileparts(fileparts(mfilename('fullpath'))),'reports')
end
if ~isfolder(outDir), mkdir(outDir); end

% --- append to CSV ------------------------------------------------------
row = table(string(record.file), string(record.timestamp), ...
            string(record.emotion), record.confidence, ...
            string(record.stress), record.stressScore, ...
    VariableNames=["File","Timestamp","Emotion","Confidence","Stress","StressScore"]);
csv = fullfile(outDir, "history.csv");
if isfile(csv)
    writetable(row, csv, WriteMode="append");
else
    writetable(row, csv);
end

% --- text card ----------------------------------------------------------
stamp = datestr(now,'yyyymmdd_HHMMSS'); %#ok<DATST>
outFile = fullfile(outDir, "report_" + stamp + ".txt");
fid = fopen(outFile, "w");
fprintf(fid, "SPEECH EMOTION & STRESS REPORT\n==============================\n");
fprintf(fid, "File       : %s\n", record.file);
fprintf(fid, "Timestamp  : %s\n", record.timestamp);
fprintf(fid, "Emotion    : %s (%.0f%%)\n", record.emotion, 100*record.confidence);
fprintf(fid, "Stress     : %s (%.2f)\n\n", record.stress, record.stressScore);
fprintf(fid, "Explanation:\n%s\n", record.reasons);
fclose(fid);
end
