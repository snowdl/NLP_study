import os
import re

def convert_absolute_to_relative(root_dir, abs_path_prefix, relative_prefix='./12_data'):
    abs_path_prefix = abs_path_prefix.rstrip('/') + '/'
    pattern = re.compile(re.escape(abs_path_prefix))

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(('.py', '.ipynb', '.txt', '.md')):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if abs_path_prefix in content:
                        new_content = pattern.sub(relative_prefix + '/', content)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f'Updated {file_path}')
                except Exception as e:
                    print(f'Skipped {file_path} due to error: {e}')

if __name__ == "__main__":
    PROJECT_ROOT = "./"
    ABSOLUTE_PREFIX = "/Users/jessicahong"
    RELATIVE_PREFIX = "./12_data"

    convert_absolute_to_relative(PROJECT_ROOT, ABSOLUTE_PREFIX, RELATIVE_PREFIX)
