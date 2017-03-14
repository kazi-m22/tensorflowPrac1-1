import random

from mnist import MNIST

mndata = MNIST('sample')

images, labels = mndata.load_training()
# or
# images, labels = mndata.load_testing()

index = random.randrange(0, len(images))  # choose an index ;-)
for i in range(100):
    print(mndata.display(images[i]))
    print ("\n")