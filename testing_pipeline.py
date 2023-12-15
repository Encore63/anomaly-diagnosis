import torch


def test(test_iter, model_path, _criterion):
    _model = torch.load(model_path)
    _model.eval()
    count, accuracy = 0, 0
    for i, (image, label) in enumerate(test_iter):
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        output = _model(image)
        loss = _criterion(output, label)
        count += torch.eq(torch.argmax(output, 1), label).sum().item()
    accuracy = count / len(test_iter)
    print('Test accuracy: {:.4f}%'.format(accuracy * 100))