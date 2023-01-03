lines = []
with open('./cuda_compile_commands.log', mode='r', encoding='utf-8') as infile:
    lines = infile.readlines()
    lines = lines[:int(len(lines) / 2)]


ans_list = []
for line in lines:
    file_path = line[len('/usr/local/cuda/bin/nvcc -M -D__CUDACC__ '):]
    file_path = file_path[:file_path.find(' ')]
    temp_dict = {
        'directory': '/harmonia/open-harmonia',
        'command': line,
        'file': file_path
    }
    ans_list.append(temp_dict)


with open('./basic_compile_commands.json', mode='r', encoding='utf-8') as infile:
    import json
    basic = json.load(infile)
    basic = list(basic)
    basic.extend(ans_list)
    with open('./compile_commands.json', mode='w+', encoding='utf-8') as outfile:
        json.dump(basic, outfile)
