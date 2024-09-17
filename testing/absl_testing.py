from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('name', default='World', help='Name to greet')

def main(argv):
    print(f'Hello, {FLAGS.name}!')

if __name__ == '__main__':
    app.run(main)
