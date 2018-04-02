import os
import subprocess

if __name__ == "__main__":
    file = open('vtk_problems.txt', 'r')

    for line in file.readlines():
        line = line.split()

        if len(line) < 2:
            continue
        library = line[3].replace('-l', '')
        current_package = os.path.join('/usr/local/lib/', 'lib' + library + '-6.3.so')
        wanted_package = os.path.join('/usr/local/lib', 'lib' + library + '.so')
        subprocess.Popen(["ln", "-s", current_package, wanted_package])
