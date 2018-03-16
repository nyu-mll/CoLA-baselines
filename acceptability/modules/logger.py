from __future__ import print_function
import os


class Logger:
    # Implement filewrite later
    def __init__(self, args):
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)

        log_file = os.path.join(args.logs_dir, args.experiment_name + '.log')

        self.log_file = open(log_file, 'a', args.buffer_size)
        self.should_log = not args.should_not_log

    def __del__(self):
        self.log_file.close()

    def write(self, x):
        # Write to log file
        if self.should_log:
            self.log_file.write(str(x) + '\n')

        # Print to stdout
        print(x)

    def write_new_line(self):
        self.write(" ")
