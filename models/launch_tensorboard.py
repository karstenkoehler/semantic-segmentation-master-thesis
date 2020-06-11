from time import sleep
from tensorboard import program

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', './unet/tf-logs'])
url = tb.launch()
print(url)

while True:
    sleep(1)