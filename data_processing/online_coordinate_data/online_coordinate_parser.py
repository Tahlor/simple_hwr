HEIGHT = 61


def test_cnn():
    import torch
    from models.basic import BidirectionalRNN, CNN
    import torch.nn as nn

    cnn = CNN(nc=1)
    pool = nn.MaxPool2d(3, (4, 1), padding=1)
    batch = 7
    y = torch.rand(batch, 1, HEIGHT, 1024)
    a, b = cnn(y, intermediate_level=13)
    new = cnn.post_process(pool(b))

    final = torch.cat([a, new], dim=2)
    print(a.size())
    print(final.size())

    for x in range(1000,1100):
        y = torch.rand(2, 1, HEIGHT, x)
        a,b = cnn(y, intermediate_level=13)

        print(a.size(), b.size())
        new = cnn.post_process(pool(b)).size()
        print(new)
        assert new == a.size()

def test_loss():
    import torch
    batch = 3
    y = torch.rand(batch, 1, HEIGHT, 1024)
    x = y
    z = l1_loss(y,x)
    print(z)

def test_stroke_parse():
    ## Add terminating stroke!!
    path = Path("../prepare_online_data/lines-xml/a01-000u-06.xml")
    x = get_gts(path, instances=30)

if __name__=="__main__":
    x = test_stroke_parse()
    print(x)

