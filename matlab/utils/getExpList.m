function expList = getExpList(cacheDir)
%GETEXPDIRS Returns a cell array of experiment results

% Get a list of the model names
files = ignoreSystemFiles(dir(cacheDir));
expNames = {files.name};

keyboard

expDirs = {};
for i = 1:numel(modelNames)
    model = modelNames{i};
    modelFiles = ignoreSystemFiles(dir(fullfile(rootDir, model)));
    modelFiles = {modelFiles.name};
    for j = 1:numel(modelFiles)
        expDirs{end + 1} = fullfile(rootDir, modelNames{i}, modelFiles{j}, 'train');
    end
end
