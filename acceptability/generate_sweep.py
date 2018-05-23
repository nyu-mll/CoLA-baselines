import argparse
import os
import hyperopt.pyll.stochastic as stoc

from copy import deepcopy
from hyperopt import hp

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--folder', default="/home/$USER/acceptability-judgments",
                    help="Path for acceptability judgments repository")
parser.add_argument('-sf', '--slurm_folder', default="./slurm_jobs",
                    help="Folder in which we should generate sbatch files")
parser.add_argument('-n', '--num_sweeps', type=int, default=1,
                    help="Number of sweeps to generate")

parser.add_argument('-j', '--job_name', default=None,
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
parser.add_argument('-eu', '--email', default=None,
                    help="Email to be mailed for slurm notification")
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

classifier_parser = subparsers.add_parser('classifier',
                            help="Generate sweeps for classifier")
classifier_parser.set_defaults(sweep_type="classifier")

classifier_parser.add_argument('-v', '--vocab', type=str, help="Vocab file location")

classifier_parser.add_argument('--encoder_path', type=str, default=None,
                                  help="Location of encoder checkpoint")
classifier_parser.add_argument('--encoding_type', type=str, default=None,
                                  help="Class of encoder")
classifier_parser.add_argument('--embedding_path', type=str, default=None,
                                  help="Path of embedding to load")
classifier_parser.add_argument('--encoding_size', type=int, default=None,
                                  help="Size of encoding, only to be used if you are loading a pretrained encoder")
classifier_parser.add_argument('--encoder_num_layers', type=int, default=None,
                                  help="Num layers of encoder, only to be used if you are loading a pretrained encoder")
classifier_parser.add_argument('--embedding_size', type=int, default=None,
                                  help="Embedding size, only to used if you are loading a pretrained encoder")
classifier_parser.add_argument('--max_pool', action="store_true", default=False,
                                  help="Use max-pooling for CBOW")
classifier_parser.add_argument('--train_embeddings', action="store_true", default=False,
                                  help="Train word embeddings")
classifier_parser.add_argument('--glove', action="store_true", default=False,
                                  help="Use glove")
classifier_parser.add_argument('--lm_path', type=str, default=None,
                                  help="LM path, to be used with elmo classifier")
classifier_parser.add_argument('-m', '--model', type=str, default=None,
                                  help="Model")
classifier_parser.add_argument("--should_not_preprocess_data", action="store_true", default=False,
                                  help="Whether to preprocess data? Default: true (Will preprocess)")
classifier_parser.add_argument("--imbalance", action="store_true", default=False,
                                  help="Is there class imbalance?")
classifier_parser.add_argument("--should_not_lowercase", action="store_true", default=False,
                                  help="Should lowercase data? Default: true (Will lowercase)")
classifier_parser.add_argument("--preprocess_tokenizer", default=None, type=str,
                                  help="Type of tokenizer to use (space|nltk)")

hashbang_line = '#!/bin/bash'

space = {
    'lm': hp.choice('lm', [{
            'hidden_size': hp.uniform('hidden_size', 300, 1200),
            'embedding_size': hp.uniform('embedding_size', 200, 600),
            'learning_rate': hp.uniform('learning_rate', -4, -2.5),
            'num_layers': hp.uniform('num_layers', 1, 5),
            'dropout': hp.choice('dropout', [0.2, 0.5])
    }]),
    'classifier': hp.choice('classifier', [{
            'hidden_size': hp.uniform('hidden_size', 20, 1200),
            'embedding_size': hp.uniform('embedding_size', 200, 600),
            'learning_rate': hp.uniform('learning_rate', -5, -3.75),
            'num_layers': hp.uniform('num_layers', 1, 5 ),
            'encoding_size': hp.uniform('encoding_size', 300, 1200),
            'encoder_num_layers': hp.uniform('encoder_num_layers', 1, 5),
            'dropout': hp.choice('dropout', [0.2, 0.5])
    }])
}

def generate_lm_sweeps(args):
    all_lines, post_shell = get_fixed_lines(args)

    run_line = get_fixed_lm_run_params(args)

    run_line = 'python -u acceptability/lm_run.py ' + run_line

    current_space = space[args.sweep_type]

    for index in range(args.num_sweeps):
        lines = deepcopy(all_lines)
        params_line, output_name = get_sampled_params_for_lm(current_space, index)

        lines[4] += str(index)

        if args.email:
            lines[9] = lines[9] + '-' + str(index) + '-%j_' + output_name
        else:
            lines[7] = lines[7] + '-' + str(index) + '-%j_' + output_name

        params_line = run_line + ' ' + params_line

        lines.append(params_line)

        lines = lines + post_shell
        slurm_file = '\n'.join(lines)

        write_slurm_file(slurm_file, args.slurm_folder, args.sweep_type, args.model, index)


def generate_classifier_sweeps(args):
    all_lines, post_shell = get_fixed_lines(args)
    run_line = get_fixed_classifier_run_params(args)

    run_line = 'python -u acceptability/run.py ' + run_line

    current_space = space[args.sweep_type]

    has_pretrained_encoder = args.encoder_path is not None
    for index in range(args.num_sweeps):
        lines = deepcopy(all_lines)

        params_line, output_name = get_sampled_params_for_classifier(args, current_space,
                                   index, has_pretrained_encoder)

        lines[4] += str(index)

        if args.email:
            lines[9] = lines[9] + '-' + str(index) + '-%j_' + output_name
        else:
            lines[7] = lines[7] + '-' + str(index) + '-%j_' + output_name

        params_line = run_line + ' ' + params_line

        lines.append(params_line)

        lines = lines + post_shell
        slurm_file = '\n'.join(lines)

        write_slurm_file(slurm_file, args.slurm_folder, args.sweep_type, args.model, index)


def get_fixed_lines(args):
    all_lines = [hashbang_line, '']

    sbatch_lines = generate_sbatch_params(args)
    module_lines = [get_module_load_lines()]
    cd_lines = ['cdir=' + args.folder, 'cd $cdir']
    pre_shell = get_shell_line(args.pre_command)
    post_shell = get_shell_line(args.post_command)
    xdg_line = [get_xdg_line()]

    all_lines = all_lines + sbatch_lines + module_lines + cd_lines + pre_shell + xdg_line

    return all_lines, post_shell

def write_slurm_file(data, folder, typ, model_name, index):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_name = 'run_acceptabilty_%s_%s_%d.sbatch' % (typ, model_name, index)
    with open(os.path.join(folder, file_name), 'w') as f:
        f.write(data)

def generate_sbatch_params(args):
    params = {
        'job-name': 'a' + args.sweep_type if args.job_name is None else args.job_name,
        'output': 'slurm',
        'nodes': 1,
        'cpus-per-task': args.cpus_per_task,
        'mem': args.mem,
        'time': args.time,
        'gres': args.gres,
    }

    if args.email:
        params['mail-type'] = 'ALL'
        params['mail-user'] = args.email

    lines = []
    sbatch_prepend = '#SBATCH '
    for key in sorted(list(params.keys())):
        lines.append('%s --%s=%s' % (sbatch_prepend, key, str(params[key])))

    return lines

def get_module_load_lines():
    return """module purge
module load cuda/8.0.44
module load cudnn/8.0v5.1
"""

def get_fixed_lm_run_params(args):
    params = ['-d', args.data, '-v', args.vocab, '--save_loc', args.save_loc,
              '--logs_dir', args.logs_dir, '-g', '-r', '-p', str(args.patience)]

    if args.stages_per_epoch is not None:
        params.append('-se')
        params.append(str(args.stages_per_epoch))

    if args.epochs is not None:
        params.append('-e')
        params.append(str(args.epochs))

    return ' '.join(params)

def get_fixed_classifier_run_params(args):
    params = ['-m', args.model, '-d', args.data, '--save_loc', args.save_loc, '--vocab_file', args.vocab,
              '--logs_dir', args.logs_dir, '-g', '-r', '-p', str(args.patience)]

    if args.max_pool:
        params.append('--max_pool')

    if args.should_not_preprocess_data:
        params.append('--should_not_preprocess_data')

    if args.should_not_lowercase:
        params.append('--should_not_lowercase')

    if args.imbalance:
        params.append('--imbalance')

    if args.train_embeddings:
        params.append('--train_embeddings')

    if args.glove:
        params.append('--glove')

    if args.preprocess_tokenizer is not None:
        params.append('--preprocess_tokenizer')
        params.append(args.preprocess_tokenizer)

    if args.stages_per_epoch is not None:
        params.append('-se')
        params.append(str(args.stages_per_epoch))

    if args.epochs is not None:
        params.append('-n')
        params.append(str(args.epochs))

    if args.encoder_path is not None:
        params.append('--encoder_path')
        params.append(str(args.encoder_path))

        if args.encoder_num_layers is not None:
            params.append('--encoder_num_layers')
            params.append(str(args.encoder_num_layers))

        if args.encoding_size is not None:
            params.append('--encoding_size')
            params.append(str(args.encoding_size))

        if args.embedding_size is not None:
            params.append('--embedding_size')
            params.append(str(args.embedding_size))

    if args.encoding_type is not None:
        params.append('--encoding_type')
        params.append(str(args.encoding_type))

    if args.embedding_path is not None:
        params.append('--embedding_path')
        params.append(str(args.embedding_path))

    if args.lm_path is not None:
        params.append('--lm_path')
        params.append(str(args.lm_path))

    return ' '.join(params)

def get_sampled_params_for_classifier(args, space, index=1, has_pretrained_encoder=False):
    sample = stoc.sample(space)
    sample['learning_rate'] = 10 ** (sample['learning_rate'])
    sample['hidden_size'] = int(sample['hidden_size'])
    sample['embedding_size'] = int(sample['embedding_size'])
    sample['num_layers'] = int(sample['num_layers'])
    sample['encoding_size'] = int(sample['encoding_size'])
    sample['encoder_num_layers'] = int(sample['encoder_num_layers'])


    output = 'lr_%.5f_nl_%d_hs_%d_do_%.1f' % (sample['learning_rate'],
             sample['num_layers'], sample['hidden_size'], sample['dropout'])

    params = '-lr %.5f -nl %d -hs %d -do %.1f' % (sample['learning_rate'],
             sample['num_layers'], sample['hidden_size'], sample['dropout'])

    if has_pretrained_encoder:
        sample.pop('encoder_num_layers')
        sample.pop('encoding_size')
        sample.pop('embedding_size')
        output += '_ed_%d_es_%d_enl_%d.out' % (args.embedding_size,
                  args.encoder_num_layers, args.encoding_size)
    else:
        output += '_ed_%d_es_%d_enl_%d.out' % (sample['embedding_size'], sample['encoding_size'],
                  sample['encoder_num_layers'])
        params += ' -es %d --encoding_size %d --encoder_num_layers %d' % (sample['embedding_size'],
                    sample['encoding_size'],
                    sample['encoder_num_layers'])

    print("Sweep ", index, sample)

    return params, output

def get_sampled_params_for_lm(space, index=1):
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
    elif args.sweep_type == 'classifier':
        generate_classifier_sweeps(args)
    else:
        print("Must select one from lm or classifier")
