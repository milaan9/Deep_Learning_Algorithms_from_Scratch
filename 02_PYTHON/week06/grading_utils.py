def test_vocab(vocab):
    return len(vocab)


def test_network(network):
    return int(all(param.grad is not None for param in network.parameters()))


def test_batch(batch):
    return int(batch[0].shape[0])
