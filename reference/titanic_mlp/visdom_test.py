from visdom import Visdom
from hwr_utils import visualize
import numpy as np

viz = Visdom(port=8080)
viz.close()

# Make a line
test = visualize.Plot("Line Test")
test.register_plot("Scatter Loss", "Epoch", "Loss")
test.register_plot("Line Loss", "Epoch", "Loss", plot_type="line")

# Use visdom directly
def use_visdom():
    test.viz.line(X=np.array([1, 2, 3]), Y=np.array([1, 5, 9]), win=test.windows["Line Loss"], update='append')

    # Make a scatter plot and update it
    x = np.array([[1, 2], [2, 4, ]])
    test.viz.scatter(X=x, win=test.windows["Scatter Loss"], update='append')

    x = np.array([[6, 2], [8, 4]])
    test.viz.scatter(X=x, win=test.windows["Scatter Loss"], update='append')

# Use my class
def use_my_visualizer():
    x = np.array([1, 2, 3])
    y = np.array([1, 5, 9])

    test.update_plot("Line Loss", x, y)
    test.update_plot("Line Loss", y, x)

    x = np.array([[1, 2], [2, 4, ]])
    test.update_plot("Scatter Loss", x=x[0], y=x[1])

    x = np.array([[6, 2], [8, 4]])
    test.update_plot("Scatter Loss", x=x[0], y=x[1])

use_my_visualizer()
