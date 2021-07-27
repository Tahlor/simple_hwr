
def load():
    with open("task/trainset.txt") as f:
        training_set = set([s.strip() for s in f.readlines()])

    with open("task/validationset1.txt") as f:
        val1_set = set([s.strip() for s in f.readlines()])

    with open("task/validationset2.txt") as f:
        val2_set = set([s.strip() for s in f.readlines()])

    with open("task/testset.txt") as f:
        test_set = set([s.strip() for s in f.readlines()])

    return training_set, val1_set, val2_set, test_set
