import glob
import os
import sys

dir = sys.argv[sys.argv.index('-d') + 1]
output_file = sys.argv[sys.argv.index('-of') + 1]
score_files = glob.iglob(os.path.join(dir, '**/av_score.txt'))
with open(output_file, mode='w') as out_f:
    for score_file in score_files:
        with open(score_file, mode='rt') as in_f:
            out_f.write(score_file)
            out_f.write('\n')
            for line in in_f.readlines():
                out_f.write(line)
                out_f.write('\n')
            out_f.write('\n')
