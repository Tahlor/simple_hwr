import numpy as np
import warnings

class Stat:
    def __init__(self, y, x, x_title="", y_title="", name="", plot=True, ymax=None, accumulator_freq=None, **kwargs):
        """

        Args:
            y (list): iterable (e.g. list) for storing y-axis values of statistic
            x (list): iterable (e.g. list) for storing x-axis values of statistic (e.g. epochs)
            x_title:
            y_title:
            name (str):
            plot (str):
            ymax (float):
            accumulator_freq: when should the variable be accumulated (e.g. each epoch, every "step", every X steps, etc.


        """
        super().__init__()
        self.y = y
        self.x = x
        self.current_weight = 0
        self.current_sum = 0
        self.accumlator_active = False
        self.updated_since_plot = False # has the value of this stat been changed since the last time it was updated?
        self.accumulator_freq = None # epoch or instances; when should this statistic accumulate?

        # Plot details
        self.x_title = x_title
        self.y_title = y_title
        self.ymax = ymax
        self.name = name
        self.plot = plot
        self.plot_update_length = 1 # add last X items from y-list to plot

    def yappend(self, new_y, new_x):
        """ Add a new y-value

        Args:
            new_y:

        Returns:

        """
        self.x.append(new_x)
        self.y.append(new_y)
        self.updated_since_plot = True

    def default(self, o):
        return o.__dict__

    def accumulate(self, sum, weight):
        if not self.current_sum >= 0:
            warnings.warn(f"{self.current_sum} should be greater than 0")
        self.current_sum += sum
        self.current_weight += weight

        if not self.accumlator_active:
            self.accumlator_active = True

    def reset_accumulator(self, new_x):
        if self.accumlator_active:

            self.y.append(self.current_sum / self.current_weight)
            self.x.append(new_x)
            self.current_weight = 0
            self.current_sum = 0
            self.accumlator_active = False
            self.updated_since_plot = True

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.__dict__)

    def get_last(self):
        if self.y:
            return self.y[-1]
        else:
            print("Stat error: No y-value yet")
            return 0

    def get_last_epoch(self):
        # ASSUMES IT'S BEING MEASURED ON EPOCH LEVEL
        try:
            x = np.array(self.x)
            y = np.array(self.y)
            x_max = x[-1]
            x_min = x[-1] - 1
            args = np.argwhere((x >= x_min) & (x <= x_max))
            return np.mean(y[args][y[args] != np.array(None)])
        except:
            print("Problem with getting value for last epoch")
            return 0


primitive = (int, str, bool, float)
def is_primitive(thing):
    return isinstance(thing, primitive)

class AutoStat(Stat):
    def __init__(self, counter_obj, x_weight, x_plot, x_title="", y_title="", name="", plot=True, ymax=None, train=True, **kwargs):
        """ AutoStat - same as Stat, but don't need to specify the x-coords every time
            Specify the x_weight once, which should be an ADDRESS of some object (not an actual number) (e.g. number of instances, number of datapoints)
            Specify the x_plot once, which should be an ADDRESS of some object (not an actual number) (e.g. number of epochs -- this will be the x-axis)

        Args:
            (x: the x plot values)
            counter_obj: A TrainingCounter object - keeps tack of number of epochs etc.
            x_weight (str): The attribute in the counter object that will be used for determining the weighting
            x_plot (str): The attribute in the counter object that will be used for determining the x-axis label
            x_title:
            y_title:
            name:
            plot:
            ymax:
            train: if the stat is a training one (the weighting is different; weighting is constant for test sets)
        """
        super().__init__(y=[None], x=[0], x_title=x_title, y_title=y_title, name=name, plot=plot, ymax=ymax)
        self.last_weight_step = 0
        self.x_counter = counter_obj
        self.x_weight = x_weight
        self.x_plot = x_plot
        self.train = train
        self.current_sum = 0

    def get_weight(self):
        if self.train:
            new_step = self.x_counter.__dict__[self.x_weight]
            weight = (new_step - self.last_weight_step)
            self.last_weight_step = new_step
        else: # if not training, the desired_num_of_strokes is constant;
            assert ("test" in self.x_weight.lower() or "valida" in self.x_weight.lower()) # make sure the key is appropriate
            weight = self.x_counter.__dict__[self.x_weight]
        if weight == 0:
            print("Error with weight - should be non-zero - using 1")
            weight = 1
        try:
            assert weight > 0
        except:
            raise Exception(f"Negative Weight: {self.__dict__}")
        return weight

    def get_x(self):
        return self.x_counter.__dict__[self.x_plot]

    def accumulate(self, sum, weight=None):
        self.current_sum += sum
        self.accumlator_active = True

    def reset_accumulator(self, new_x=None):
        if self.accumlator_active:
            weight = self.get_weight()
            # Update plot values
            self.y.append(self.current_sum / weight)
            self.x.append(self.get_x())

            # Reset Accumulator
            self.current_sum = 0
            self.accumlator_active = False
            self.updated_since_plot = True


class Counter:
    def __init__(self, instances_per_epoch=1, epochs=0, updates=0, instances=0, training_pred_count=0, test_instances=1,
                 test_pred_length_static=1, test_pred_count=0, validation_pred_count=0):
        """

        Args:
            instances_per_epoch:
            epochs:
            updates:
            instances:
            training_pred_count: Running count of the number of individual predictions (i.e. stroke points) in training data
            test_instances: Size of test data (instances)
            test_pred_length_static: Total number of predictions (i.e. stroke points) in test data
        """

        self.epochs = epochs
        self.updates = updates
        self.instances = instances
        self.instances_per_epoch = instances_per_epoch

        if self.instances_per_epoch: # if there are training instances
            self.epoch_decimal = self.instances/self.instances_per_epoch
        else:
            self.epoch_decimal = 0

        self.training_pred_count = training_pred_count

        self.test_instances = test_instances
        self.test_pred_length_static = test_pred_length_static # If this is not constant, put it in train mode!
        self.test_pred_count = test_pred_count
        self.validation_pred_count = validation_pred_count

    def update(self, epochs=0, instances=0, updates=0, training_pred_count=0, test_pred_count=0, validation_pred_count=0):
        self.epochs += epochs
        self.instances += instances
        self.updates += updates
        self.epoch_decimal = self.instances / self.instances_per_epoch

        # Both are same as above, always incrementing
        self.training_pred_count += training_pred_count
        self.test_pred_count += test_pred_count
        self.validation_pred_count += validation_pred_count

if __name__=='__main__':
    training_counter = Counter()
    training_counter.epochs += 1
    print(training_counter.epochs)
    #AutoStat()

