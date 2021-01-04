import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import datasets
import models
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def train_loop(model, criterion1, criterion2, epochs, data_loader, content_embed, class_embed,
               batch_size, tf_writer):
    flag = True
    ep = 0
    class_dim, content_dim = class_embed.shape[1], content_embed.shape[1]
    content_tensor = torch.zeros((batch_size, content_dim), requires_grad=True)
    class_tensor = torch.zeros((batch_size, class_dim), requires_grad=True)
    optimizer = optim.Adam([{'params': model.parameters()},
                            {'params': content_tensor, 'weight_decay': 0.001},
                            {'params': class_tensor}])
    running_loss = 0.0
    im_batch = torch.zeros((int(epochs/2), 1, 28, 28))
    for epoch in range(epochs):
        print("epoch: %d" % ep)
        ep += 1
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            optimizer.zero_grad()
            images, labels, ids = data
            content_tensor.data = content_embed[ids]
            class_tensor.data = class_embed[labels]
            noise = torch.normal(0, 0.3, size=(batch_size, content_tensor.shape[1]))
            content_tensor.data += noise
            input_tensor = torch.cat((content_tensor, class_tensor), dim=1)
            outputs = model(input_tensor)
            loss = criterion1(outputs, images) + criterion2(outputs, images)
            loss.backward()
            optimizer.step()
            class_embed[labels] = class_tensor  # - noise ?
            content_embed[ids] = content_tensor
            running_loss += loss.item()
            if i % 500 == 499:  # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                if epoch % 2 == 0:
                    im_batch[int(epoch/2)] = outputs[2].data[0].squeeze()
                    if flag:
                        tf_writer.add_image('orig_im', images[2], 0)
                        flag = False
                tf_writer.add_scalar('Loss/Loss_vs_Epochs', running_loss / 500, ep)

    tf_writer.add_images('my_image_batch', im_batch, 0)
    print('Finished Training')
    return running_loss / 500


def embed_data(to_embed, dim=50):
    emb = nn.Embedding(len(to_embed)+1, dim)
    return emb(to_embed)


def prepare_data_embedding(data, class_dim, content_dim, n=2000,):
    classes = torch.LongTensor(range(11))
    classes = embed_data(classes, class_dim)
    ids = torch.LongTensor(range(n))
    ids = embed_data(ids, content_dim)
    return classes, ids


def predict_examples(data, model, content_code, class_code):
    r = 3
    c = 4
    fig, axs = plt.subplots(r, c)
    fig.suptitle('Examples of model input vs output')
    for i in range(r):
        for j in range(c):
            if j % 2 == 0:
                idx = np.random.randint(0, 2000)
                im, label, id = data.__getitem__(idx)
                im = im.squeeze()
                res = single_sample_pred(model, content_code[id], class_code[label])
                res = res.data[0].squeeze()
                axs[i, j].imshow(im)
                axs[i, j+1].imshow(res)
                if i == 0:
                    axs[i, j].set_title("Original")
                    axs[i, j+1].set_title("Predicted")
    plt.show()


def single_sample_pred(model, content_code, class_code):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.cat((content_code, class_code))
        res = model(input_tensor)
    return res


def predict_with_wrong_class(data, model, content_code, class_code, tf_writer):
    idx = np.random.randint(0, 2000)
    model.eval()
    im, label, id = data.__getitem__(idx)
    im_batch = torch.zeros((int(11), 1, 28, 28))
    im_batch[0] = torch.from_numpy(im.squeeze())
    for i in range(10):
        res = single_sample_pred(model, content_code[id], class_code[i])
        im_batch[i+1] = res.data[0].squeeze()
    tf_writer.add_images('my_image_batch', im_batch, 0)


def use_saved_model():
    writer = SummaryWriter()
    data = datasets.MNIST(2000)
    content_code_dim = 40
    class_code_dim = 10
    content_code = torch.load('./data/reg_content_code.pt')
    class_code = torch.load('./data/reg_class_code.pt')
    generator = models.GeneratorForMnistGLO(class_code_dim + content_code_dim)
    generator.load("./data/reg_trained_model.ckpt")
    # predict_examples(data, generator, content_code, class_code)
    # predict_with_wrong_class(data, generator, content_code, class_code, writer)
    random_code_from_trained_dist(content_code, class_code, generator, data, writer)
    writer.close()


def random_code_from_trained_dist(content_code, class_code, model, data, tf_writer):
    u = torch.mean(content_code, dim=1)
    sig = torch.std(content_code, dim=1)
    im_batch = torch.zeros((10, 1, 28, 28))
    for i in range(10):
        new_code = torch.normal(u.data[i], sig.data[i], size=(1, content_code.shape[1]))
        # new_code = torch.normal(0, 1, size=(1, content_code.shape[1]))
        pred = single_sample_pred(model, new_code[0], class_code[i])
        im_batch[i] = pred.data[0].squeeze()
        # data.present(pred)
        # random_code = torch.normal(0, 1, size=(1, content_code.shape[1]))
        # pred = single_sample_pred(model, random_code[0], class_code[7])
        # data.present(pred)
    tf_writer.add_images('random latent code from trained dist', im_batch, 0)


def weight_decay_tune(model, criterion1, criterion2, epochs, data_loader, content_embed, class_embed,
                      batch_size, tf_writer):
    decay = [0.2, 0.1, 0.001, 0.0001, 0.00001]
    min_loss = 100
    winner = 0
    for i, dec in enumerate(decay):
        loss = train_loop(model, criterion1, criterion2, epochs, data_loader, content_embed,
                          class_embed, batch_size, tf_writer)
        if loss < min_loss:
            winner = i
    print('best decay weigth is: ' + str(decay[winner]))


def train():
    writer = SummaryWriter()
    batch_size = 4
    content_code_dim = 45
    class_code_dim = 5
    data = datasets.MNIST(2000)
    # weights = [0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
    losses = []
    # for i in range(len(weights)):
    class_embed, content_embed = prepare_data_embedding(data, class_code_dim, content_code_dim, 2000)
    generator = models.GeneratorForMnistGLO(class_code_dim + content_code_dim)
    criterionL1 = nn.L1Loss()
    criterionL2 = nn.MSELoss()
    generator.train()
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    loss = train_loop(generator, criterionL1, criterionL2, 50, train_loader, content_embed,
                      class_embed, batch_size, writer)
    generator.save("./data/reg_trained_model.ckpt")
    torch.save(content_embed, './data/reg_content_code.pt')
    torch.save(class_embed, './data/reg_class_code.pt')
    predict_examples(data, generator, content_embed, class_embed)
    writer.close()


train()
# use_saved_model()
