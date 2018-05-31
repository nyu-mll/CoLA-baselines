import sys

inpfile = sys.argv[1]

with open(inpfile, 'r') as f:
    lines = f.readlines()
    final = []
    for idx, line in enumerate(lines):
        line = line.strip().split('\t')
        text = line[-1]
        final.append(str(idx + 1) + '\t' + text)

    outfile = '.'.join(inpfile.split('.')[:-1]) + '-stripped.tsv'
    final = ['Id\tSentence'] + final    
    with open(outfile, 'w') as o:
        o.write('\n'.join(final))
        
