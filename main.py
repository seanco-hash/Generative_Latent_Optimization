import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import datasets
import models
import torch.nn as nn
import matplotlib as plt


def train_loop(model, criterion1, criterion2, epochs, data_loader, content_embed, class_embed,
               batch_size):
    ep = 0
    class_dim, content_dim = class_embed.shape[1], content_embed.shape[1]
    input_tensor = torch.zeros((batch_size, content_dim + class_dim), requires_grad=True)
    optimizer = optim.Adam([{'params': model.parameters()}, {'params': input_tensor}])
    for epoch in range(epochs):
        print("epoch: %d" % ep)
        ep += 1
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            images, labels, ids = data
            inputs = content_embed[ids]
            noise = torch.normal(0, 0.3, size=(batch_size, inputs.shape[1]))
            inputs += noise
            input_tensor.data = torch.cat((inputs, class_embed[labels]), dim=1)
            # for label in labels:
            #     count[label] += 1
            optimizer.zero_grad()
            outputs = model(input_tensor)
            loss = criterion1(outputs, images) + criterion2(outputs, images)
            loss.backward()
            optimizer.step()
            class_embed[labels] = input_tensor.data[:, -class_dim::]  # - noise ?
            content_embed[ids] = input_tensor.data[:, : content_dim]
            running_loss += loss.item()
            if i % 500 == 499:  # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    # print(count)
    print('Finished Training')


def embed_data(to_embed, dim=50):
    emb = nn.Embedding(len(to_embed)+1, dim)
    return emb(to_embed)


def prepare_data_embedding(data, class_dim, content_dim, n=2000,):
    classes = torch.LongTensor(range(11))
    classes = embed_data(classes, class_dim)
    ids = torch.LongTensor(range(n))
    ids = embed_data(ids, content_dim)
    # train_data = []
    # for i in range(n):
    #     train_data.append((ids[i], data.__getitem__(i)[1]))

    # return classes, np.array(train_data)
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
                res = single_sample_pred(model, im, content_code[id], class_code[label])
                res = res.data[1].squeeze()
                axs[i, j].imshow(im)
                axs[i, j+1].imshow(res)
                axs[i, j].set_title("Original id: " + str(id))
                axs[i, j+1].set_title("Predicted id: " + str(id))
    plt.show()


def single_sample_pred(model, im, content_code, class_code):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.cat((content_code, class_code))
        res = model(input_tensor)
    return res


def train():
    batch_size = 4
    content_code_dim = 50
    class_code_dim = 3
    data = datasets.MNIST(2000)
    class_embed, content_embed = prepare_data_embedding(data, class_code_dim, content_code_dim, 2000)
    generator = models.GeneratorForMnistGLO(class_code_dim + content_code_dim)
    criterionL1 = nn.L1Loss()
    criterionL2 = nn.MSELoss()
    generator.train()
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loop(generator, criterionL1, criterionL2, 2, train_loader, content_embed, class_embed,
               batch_size)
    generator.save("./data/my_trained_model.ckpt")
    torch.save(content_embed, './data/content_code.pt')
    torch.save(class_embed, './data/class_code.pt')
    predict_examples(data, generator, content_embed, class_embed)


train()
