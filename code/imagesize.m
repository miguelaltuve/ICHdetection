

% data path
cd C:\Users\migue\Dropbox\publicaciones\HIC_detection\data\healthy

FileName = dir('*.png');


for i = 1:size(FileName,1)

    A = imread(FileName(i).name);

    %     B = A(:,:,1)-A(:,:,2)-A(:,:,3);
    Size1(i,:) = size (A);
    %     figure, imshow(A)
    %     figure, imshow(B);
    %     min(min(B))
    %     max(max(B))
    %     clc
    %     cclose all;
end

cd C:\Users\migue\Dropbox\publicaciones\HIC_detection\data\ICH\
FileName = dir('*.png');


for i = 1:size(FileName,1)

    A = imread(FileName(i).name);
    Size2(i,:) = size (A);

end