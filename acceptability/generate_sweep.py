import argparse
import os
import hyperopt.pyll.stochastic as stoc

from copy import deepcopy
from hyperopt import hp

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--folder', default="/home/$USER/acceptability-judgments",
                    help="Path for acceptability judgments repository")
parser.add_argument('-n', '--num_sweeps', type=int, default=1,
                    help="Number of sweeps to generate")

parser.add_argument('-j', '--job_name', default="myJob",
                    help="Job name, sweep sample number will be appended to this")
parser.add_argument('-t', '--time', default="47:00:00",
                    help="Time limit of sweep")
parser.add_argument('-me', '--mem', default="32GB",
                    help="Memory for sweep")
parser.add_argument('-g', '--gres', default="gpu:1",
                    help="GPU type to be specified in sweep")
parser.add_argument('-c', '--cpus-per-task', default="2",
                    help="CPUs per task to be specified in sweeps")
parser.add_argument('-p', '--patience', type=int, default=4, help="Early stopping patience")
parser.add_argument('-l', '--logs_dir', default='./logs',
                    help="Directory for storing logs")
parser.add_argument('-s', '--save_loc', default='./save',
                    help="Directory for saving models")
parser.add_argument('-e', '--epochs', type=int, default=None,
                    help="Epochs")
parser.add_argument('-d', '--data', default='./data',
                    help="Folder containing data tsvs")
parser.add_argument('-pr', '--pre_command', default=None,
                    help="Shell command to run before running main command")
parser.add_argument('-ps', '--post_command', default=None,
                    help="Shell command to run after running main command")
parser.add_argument('-se', '--stages_per_epoch', type=int, default=None,
                    help="Number of evaluation steps, if not passed default will be used")


subparsers = parser.add_subparsers()
lm_parser = subparsers.add_parser('lm', help="Generate sweeps for lm")

lm_parser.add_argument('-v', '--vocab', help="Vocab file location")
lm_parser.add_argument('-m', '--model', default="lstm",
                       help="Model type to be used for lm")
lm_parser.set_defaults(sweep_type="lm")


hashbang_line = '#!/bin/bash'

space = {
    'lm': {
        'lstm': hp.choice('lstm', [{
            'hidden_size': hp.uniform('hidden_size', 300, 1200),
            'embedding_size': hp.uniform('embedding_size', 200, 600),
            'learning_rate': hp.choice('learning_rate', [-3, -4]),
            'num_layers': hp.uniform('num_layers', 1, 4),
            'dropout': hp.choice('dropout', [0.2, 0.5])
        }])
    }
}

def generate_lm_sweeps(args):
    all_lines = [hashbang_line, '']

    sbatch_lines = generate_sbatch_params(args)
    module_lines = [get_module_load_lines()]
    cd_lines = ['cdir=' + args.folder, 'cd $cdir']
    pre_shell = get_shell_line(args.pre_command)
    post_shell = get_shell_line(args.post_command)
    xdg_line = [get_xdg_line()]

    all_lines = all_lines + sbatch_lines + module_lines + cd_lines + pre_shell + xdg_line
    run_line = get_fixed_lm_run_params(args)

    run_line = 'python -u acceptability/lm_run.py ' + run_line

    current_space = space['lm'][args.model]

    for index in range(args.num_sweeps):
        lines = deepcopy(all_lines)
        params_line, output_name = get_sampled_params(current_space, index)

        lines[2] += str(index)
        lines[3] += output_name

        params_line = run_line + ' ' + params_line

        lines.append(params_line)

        lines = lines + post_shell
        slurm_file = '\n'.join(lines)

        write_slurm_file(slurm_file, 'slurm_jobs', 'lm', args.model, index)


def generate_acceptability_sweeps(args):
    return

def write_slurm_file(data, folder, typ, model_name, index):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_name = 'run_acceptabilty_%s_%s_%d.sbatch' % (typ, model_name, index)
    with open(os.path.join(folder, file_name), 'w') as f:
        f.write(data)

def generate_sbatch_params(args):
    params = {
        'job-name': args.job_name,
        'output': 'slurm-%j_',
        'nodes': 1,
        'cpus-per-task': args.cpus_per_task,
        'mem': args.mem,
        'time': args.time,
        'gres': args.gres
    }

    lines = []
    sbatch_prepend = '#SBATCH '
    for key in params.keys():
        lines.append('%s --%s=%s' % (sbatch_prepend, key, str(params[key])))

    return lines

def get_module_load_lines():
    return """module purge
module load cuda/8.0.44
module load cudnn/8.0v5.1
"""

def get_fixed_lm_run_params(args):
    params = ['-d', args.data, '-v', args.vocab, '--save_loc', args.save_loc,
              '--logs_dir', args.logs_dir, '-g', '-p', str(args.patience)]

    if args.stages_per_epoch is not None:
        params.append('-se')
        params.append(str(args.stages_per_epoch))

    if args.epochs is not None:
        params.append('-e')
        params.append(str(args.epochs))

    return ' '.join(params)

def get_sampled_params(space, index=1):
    sample = stoc.sample(space)
    sample['learning_rate'] = 10 ** (sample['learning_rate'])
    sample['embedding_size'] = int(sample['embedding_size'])
    sample['hidden_size'] = int(sample['hidden_size'])
    sample['num_layers'] = int(sample['num_layers'])

    print("Sweep ", index, sample)

    output = 'lr_%.5f_do_%.1f_nl_%d_hs_%d_es_%d.out' % (sample['learning_rate'],
             sample['dropout'], sample['num_layers'], sample['hidden_size'],
             sample['embedding_size'])

    params = '-lr %.5f -do %.1f -nl %d -hs %d -es %d' % (sample['learning_rate'],
             sample['dropout'], sample['num_layers'], sample['hidden_size'],
             sample['embedding_size'])

    return params, output

def get_xdg_line():
    return """cat<<EOF

Running Acceptability task
EOF
unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi
"""

def get_shell_line(script):
    if script == None:
        return ['']
    else:
        return [script]


if __name__ == '__main__':
    args = parser.parse_args()

    if args.sweep_type == 'lm':
        generate_lm_sweeps(args)
    elif args.sweep_type == 'acceptability':
        generate_acceptability_sweeps(args)
