function [chan_flag_type]=chan_input(chan_flag_type)

% 定义要替换的文件名和新的flag_type值
file_name = 'Channel Model.txt'; % 文件名
% new_flag_type = 3; % 新的flag_type值

% 读取文件内容
fileID = fopen(file_name, 'r');
file_content = textscan(fileID, '%s', 'Delimiter', '\n');
fclose(fileID);

% 获取第四行的内容（假设这是flag_type所在的行）
line_to_replace = file_content{1}{4};

% 使用新的flag_type值替换行中的数字
new_line = regexprep(line_to_replace, '\d+', num2str(chan_flag_type));

% 更新文件内容
file_content{1}{4} = new_line;

% 打开文件以写入新的内容
fileID = fopen(file_name, 'w');
for i = 1:length(file_content{1})
    fprintf(fileID, '%s\n', file_content{1}{i});
end
fclose(fileID);

end





