import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd



testdf = pd.read_pickle(r"dataset.pk1")


image = np.array( testdf['Image'][0])
plt.imshow(image, cmap="gray")
plt.title("Image Test")
plt.show()
