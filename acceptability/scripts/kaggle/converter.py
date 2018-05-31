import sys

inpfile = sys.argv[1]

with open(inpfile, 'r') as f:
    lines = f.readlines()
    final = []
    for idx, line in enumerate(lines):
        line = line.strip().split('\t')
        label = line[1]
        final.append(str(idx + 1) + ',' + str(label))

    outfile = '.'.join(inpfile.split('.')[:-1]) + '.csv'
    final = ['Id,Expected'] + final    
    with open(outfile, 'w') as o:
        o.write('\n'.join(final))
        
