"""A simple example of using absl.flags and absl.app."""

from absl import app
from absl import flags

# Define a command-line flag
FLAGS = flags.FLAGS
flags.DEFINE_string('name', default='World', help='Name to greet')

# Define the main function
def main(argv):
    print(f'Hello, {FLAGS.name}!')

# Call the app.run() function
if __name__ == '__main__':
    app.run(main)
    
# Output:
# $ python absl_testing.py
# Hello, World!

# $ python absl_testing.py --name=John
# Hello, John!
