#import argparse
import pathlib

# Truncate the texts within the input_dir to texts with 2100 words and save them in output_dir
def truncate_texts(input_dir_path, output_dir_path):
    input_path = pathlib.Path(input_dir_path)
    
    #print(list(input_path.rglob('*.txt')))
    for file_path in list(input_path.rglob('*.txt')):
        #print(str(file_path).split('\\')[-1])
        
        with open(file_path, 'rt', encoding = 'UTF-8') as file:
            #print(file.name.split('\\')[-1])
            
            text = file.read()
            new_text = ''
            
            word = ''
            words_num = 0
            i = 0
            while i < len(text) and words_num < 2100:
                char = text[i]
                
                if (char >= 'a' and char <= 'z') or (char >= 'A' and char <= 'Z') or (char == '-' and word != ''):
                    word += char
                else:
                    if word != '':
                        #print(word)
                        
                        words_num += 1
                        word = ''
                
                new_text += char
                i += 1
        
        with open(output_dir_path + '\\' + file.name.split('\\')[-1], 'wt', encoding = 'UTF-8') as new_file:
            new_file.write(new_text)

# Convert the texts within the input_dir to texts in BRAT format and save them in output_dir
def convert_texts(input_dir_path, output_dir_path):
    input_path = pathlib.Path(input_dir_path)
    output_path = pathlib.Path(output_dir_path)
    for file_path in list(input_path.rglob('*.txt')):
        with open(file_path, 'rt', encoding = 'UTF-8') as read_file:
            file_name = (pathlib.Path(read_file.name).name)[0 : -4] + '_brat.txt'
            with open(pathlib.Path(output_path, file_name), 'wt', encoding = 'UTF-8') as write_file:
                for line in read_file:
                    i = 0
                    while i < len(line):
                        if line[i] != '\n':
                            if line[i] == ',' or line[i] == ';' or line[i] == ':':
                                if line[i - 1] != ' ':
                                    write_file.write(' ')
                                if i + 1 == len(line):
                                    write_file.write(line[i] + ' ')
                                else:
                                    if line[i + 1] == ' ':
                                        write_file.write(line[i] + ' ')
                                        i = i + 1
                                    else:
                                        if line[i + 1] == '-' or line[i + 1] == '"':
                                            write_file.write(line[i])
                                        else:
                                            write_file.write(line[i] + ' ')
                            elif line[i] == '.' or line[i] == '?' or line[i] == '!':
                                if line[i - 1] != ' ' and line[i - 1] != '!' and line[i - 1] != '?' and line[i - 1] != '.':
                                    write_file.write(' ')
                                if line[i] == '.':
                                    if i + 1 != len(line):
                                        if line[i + 1] == '.':
                                            write_file.write(line[i] + '.')
                                            if i + 2 != len(line):
                                                if line[i + 2] == '.':
                                                    write_file.write('.')
                                                    if i + 3 != len(line):
                                                        if line[i + 3] == '.':
                                                            write_file.write('. ')
                                                            i = i + 3
                                                        else:
                                                            write_file.write(' ')
                                                            i = i + 2
                                                    else:
                                                        write_file.write(' ')
                                                        i = i + 2
                                                else:
                                                    write_file.write(' ')
                                                    i = i + 1
                                            else:
                                                write_file.write(' ')
                                                i = i + 1
                                        else:
                                            if line[i + 1] == '"':
                                                write_file.write(line[i] + ' "\n')
                                                i = i + 1
                                            else:
                                                if line[i + 1] == '_' or line[i + 1] == '»':
                                                    if i + 2 != len(line):
                                                        if line[i + 2] == '"':
                                                            write_file.write(line[i] + ' "\n')
                                                            i = i + 2
                                                        else:
                                                            write_file.write(line[i] + '\n')
                                                            i = i + 1
                                                    else:
                                                        write_file.write(line[i] + '\n')
                                                else:
                                                    write_file.write(line[i] + '\n')
                                    else:
                                        write_file.write(line[i] + '\n')
                                else:
                                    if i + 1 != len(line):
                                        if line[i + 1] == '"':
                                            write_file.write(line[i] + ' "\n')
                                            i = i + 1
                                        else:
                                            if line[i + 1] == '_' or line[i + 1] == '»':
                                                if i + 2 != len(line):
                                                    if line[i + 2] == '"':
                                                        write_file.write(line[i] + ' "\n')
                                                        i = i + 2
                                                    else:
                                                        write_file.write(line[i] + '\n')
                                                        i = i + 1
                                                else:
                                                    write_file.write(line[i] + '\n')
                                            else:
                                                write_file.write(line[i] + '\n')
                                    else:
                                        write_file.write(line[i] + '\n')
                                if i + 1 != len(line):
                                    if line[i + 1] == ' ':
                                        i = i + 1
                            elif line[i] == '-':
                                if i > 0:
                                    if line[i - 1] != ' ' and line[i - 1] != '!' and line[i - 1] != '?' and line[i - 1] != '.':
                                        write_file.write(' ')
                                if i + 1 == len(line):
                                    write_file.write(line[i] + ' ')
                                else:
                                    if line[i + 1] == ' ':
                                        write_file.write(line[i] + ' ')
                                        i = i + 1
                                    else:
                                        if line[i + 1] == '-':
                                            if i + 2 != len(line):
                                                if line[i + 2] == ' ':
                                                    write_file.write(line[i] + '- ')
                                                    i = i + 2
                                                else:
                                                    write_file.write(line[i] + '- ')
                                                    i = i + 1
                                            else:
                                                write_file.write(line[i] + '- ')
                                        else:
                                            write_file.write(line[i] + ' ')
                            elif line[i] == '"':
                                if i > 0:
                                    if line[i - 1] != ' ' and line[i - 1] != '!' and line[i - 1] != '?' and line[i - 1] != '.':
                                        write_file.write(' ')
                                if i + 1 == len(line):
                                    write_file.write(line[i] + ' ')
                                else:
                                    if line[i + 1] == ' ':
                                        write_file.write(line[i] + ' ')
                                        i = i + 1
                                    else:
                                        if line[i + 1] == '-' or line[i + 1] == ',' or line[i + 1] == ':' or line[i + 1] == ';':
                                            write_file.write(line[i])
                                        else:
                                            write_file.write(line[i] + ' ')
                            elif line[i] == "'":
                                if i + 1 < len(line):
                                    if line[i + 1] == ' ':
                                        write_file.write(line[i] + ' ')
                                        i = i + 1
                                    else:
                                        write_file.write(line[i] + ' ')
                            elif line[i] == ' ':
                                if i == 0:
                                    while line[i] == ' ':
                                        i = i + 1
                                    if line[i] != '\n' and line[i] != '_' and line[i] != '«' and line[i] != '»' and line[i] != '*':
                                        write_file.write(line[i])
                                else:
                                    if i + 1 != len(line):
                                        if line[i + 1] != '*':
                                            write_file.write(line[i])
                            else:
                                if i + 1 != len(line):
                                    if line[i + 1] == '\n' and line[i] != '_' and line[i] != '«' and line[i] != '»' and line[i] != '*':
                                        write_file.write(line[i] + ' ')
                                    else:
                                        if line[i] != '_' and line[i] != '«' and line[i] != '»' and line[i] != '*':
                                            write_file.write(line[i])
                        i = i + 1

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input', help = 'folder where the texts to truncate are located', required = True)
    parser.add_argument('-o', '--output', help = 'folder where the texts truncated will be located', required = True)
    
    args = vars(parser.parse_args())
    
    truncate_texts(args['input'], args['output'])

    parser.add_argument('-i', '--input', help = 'folder where the texts to convert in BRAT format are located', required = True)
    parser.add_argument('-o', '--output', help = 'folder where the texts converted in BRAT format will be located', required = True)
    
    args = vars(parser.parse_args())
    
    convert_texts(args['input'], args['output'])
'''