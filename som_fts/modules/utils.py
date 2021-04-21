import os


def project_path(filepath: str = '') -> str:
    base = os.getcwd()
    PROJECT_NAME = 'Appliance-Energy-Prediction'
    if PROJECT_NAME in base:
        i = base.index(PROJECT_NAME)
        base = base[:i]
        path = base + PROJECT_NAME
    else:
        separator = '\\' if '\\' in base else '/'
        directories = base.split(separator)
        directories.reverse()
        for dir in directories:
            i = base.index(dir)
            list_dir = os.listdir(base)
            if PROJECT_NAME in list_dir:
                if base[-1] != separator:
                    base += separator
                base += PROJECT_NAME
                break
            base = base[:i]
        if PROJECT_NAME not in base:
            path = input('Project not found. Please, type the file location: ')

    path = os.path.join(path, filepath)
    return path