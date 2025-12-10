import sys, os, time

# log recorder
class Logger(object):

    def __init__(self,output_dir,stream=sys.stdout):
        log_name_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        log_name = log_name_time + ".txt"
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass