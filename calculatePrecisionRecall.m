function [precision,recall]=calculatePrecisionRecall(queryFeat, targetFeat, queryLabels, targetLabels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Description
%this function is to calculate the precision recall for NIPS2012 paper

%Input
%  queryFeat     nquery*dim feature matrix 
%  targetset     ntarget*dim feature matrix
%  queryLabels       nquery*dimLabel label matrix
%  targetLabels      ntarget*dimLabel label matrix

%Output
%  precision   recall
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nquery, dim] = size(queryFeat);
[ntarget, dim] = size(targetFeat);

% first compute the cos similarity scores, nquery x ntarget
Dist = CosineDist(queryFeat, targetFeat);

[asDist index] = sort(Dist, 2, 'ascend');
[sorted_sims, locs] = sort(Dist, 'descend');

for n = 1 : nquery
    pred_ids = index(n, :);
    relevant_ids = getRelevantID(queryLabels(n, :), targetLabels);
    
    num_relevant_images = numel(relevant_ids);
    
    pred_locations = arrayfun(@(x) find(locs == x, 1), relevant_ids);
    pred_locations_sorted = sort(pred_locations);
    
    precision = (1:num_relevant_images) ./ pred_locations_sorted;
    recall = (1:num_relevant_images) / num_relevant_images;
    
    
end

end


function ids = getRelevantID(queryVector, targetMatrix)
% given a multi-label binary vector, find its relevant index (ID) in target
% matrix
% queryVector 1xdim
% targetMatrix ntarget x dim

[ntarget, dim] = size(targetMatrix);

scores = sum(repmat(queryVector, ntarget, 1) .* targetMatrix, 2);

ids = find(scores ~= 0);

end











% 
% 
% 
% % https://stackoverflow.com/questions/25799107/how-do-i-plot-precision-recall-graphs-for-content-based-image-retrieval-in-matla/25811041#25811041
% % Let's say I have 5 images in a database of 20, 
% % and I have a bunch of similarity values between them and a query image:
% % rng(123); %// Set seed for reproducibility
% num_images = 20;
% sims = rand(1,num_images);
% 
% % Also, I know that images [1 5 7 9 12] are my relevant images.
% relevant_IDs = [1 5 7 9 12];
% num_relevant_images = numel(relevant_IDs);
% 
% % Now let's sort the similarity values in descending order, as higher values mean higher similarity. 
% % You'd reverse this if you were calculating a dissimilarity measure:
% [sorted_sims, locs] = sort(sims, 'descend');
% 
% locations_final = arrayfun(@(x) find(locs == x, 1), relevant_IDs);
% 
% % Let's sort these to get a better understand of what this is saying:
% locations_sorted = sort(locations_final);
% 
% % calculate precision and recall
% precision = (1:num_relevant_images) ./ locations_sorted;
% recall = (1:num_relevant_images) / num_relevant_images;
% 
% % Your Precision-Recall graph
% plot(recall, precision, 'b.-');
% xlabel('Recall');
% ylabel('Precision');
% title('Precision-Recall Graph - Toy Example');
% axis([0 1 0 1.05]); %// Adjust axes for better viewing
% grid;