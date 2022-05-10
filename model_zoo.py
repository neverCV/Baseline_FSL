import paddle
import paddle.nn as nn
from paddle.nn.utils import weight_norm
import numpy as np
from tqdm import tqdm
import paddle.optimizer as pdoptimizer
from paddlefsl import backbones as pdbackbones
import os
import logging

np.random.seed(10)
paddle.seed(10)

class get_log():
    """
    数据记录
    """

    def __init__(self, name, filename):
        """
        初始化
        :param name: log的名字
        :param filename: 存储的file名字、
        """
        self.log = logging.getLogger(name)
        self.log.addHandler(logging.FileHandler(filename))
        self.log.setLevel(logging.DEBUG)

    def log_trainmessage(self, epoch, trainacc, trainloss, testacc, testloss):
        """
        记录下当前epoch的信息
        :param epoch: 当前epoch数
        :param trainacc: 当前epoch训练精度
        :param trainloss: 当前epoch训练损失
        :param testacc: 当前epoch测试精度
        :param testloss: 当前epoch测试损失
        :return: 无返回
        """
        message = f'epoch:{epoch} | trainacc :{trainacc:.4f | trainloss:{trainloss:.4f} | testacc:{testacc:.4f} | testloss:{testloss:.4f}}'
        self.log_something(message)

    def log_something(self, message):
        """
        记录信息
        :param message: 用户自定义写好的字符串
        :return:
        """
        self.log.info(message)


class distLinear(nn.Layer):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias_attr=False)
        self.class_wise_learnable_norm = True  # See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            weight_norm(self.L, 'weight', dim=0)  # split the weight update component to direction and nor
            self.scale_factor = 2  # a fixed scale factor to scale the output of cos value into a reasonably large input for softmax

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x_norm = paddle.norm(x, p=2, axis=1).unsqueeze(1).expand_as(x)
        x_normalized = x.divide(x_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        # matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise
        # learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor * (cos_dist)

        return scores


class BaselineFinetune(nn.Layer):
    def __init__(self, model_func, n_way, n_support, n_query=16, loss_type="softmax"):
        super(BaselineFinetune, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query  # (change depends on input)
        self.feature = model_func
        self.feat_dim = 1600
        self.loss_type = loss_type

    def parse_feature(self, x):

        z_all = x
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query

    def set_forward(self, x, is_feature=True):
        return self.set_forward_adaptation(x, is_feature)  # Baseline always do adaptation

    def set_forward_adaptation(self, x, is_feature=True):
        assert is_feature == True, 'Baseline only support testing with feature'
        # y_support = x[1]
        # x = x[0]
        # z_support = x[:self.n_way * self.n_support]
        # z_query = x[self.n_way * self.n_support:]

        z_support, z_query = self.parse_feature(x)
        z_support = z_support.reshape([self.n_way * self.n_support, -1])
        z_query = z_query.reshape([self.n_way * self.n_query, -1])
        y_support = paddle.to_tensor(np.repeat(range(self.n_way), self.n_support), dtype=paddle.int64)

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist':
            linear_clf = distLinear(self.feat_dim, self.n_way)

        set_optimizer = pdoptimizer.Momentum(parameters=linear_clf.parameters(), learning_rate=0.01,
                                             momentum=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.clear_grad()
                selected_id = paddle.to_tensor(rand_id[i: min(i + batch_size, support_size)])
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]

                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()
        scores = linear_clf(z_query)
        return scores

    def set_forward_loss(self, x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')

    def forward(self, x):
        out = self.feature(x)
        return out


class BaselineTrain(nn.Layer):
    def __init__(self, model_func, num_class, loss_type='softmax'):
        super(BaselineTrain, self).__init__()
        self.feature = model_func  # ()
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist':  # Baseline ++
            self.classifier = distLinear(1600, num_class)
        self.loss_type = loss_type  # 'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        scores = self.forward(x)
        return self.loss_fn(scores, y)

    def train_loop(self, epoch, train_loader, optimizer):
        avg_loss = 0
        for x, y in tqdm(train_loader, desc='train ' + str(epoch), leave=False):
            optimizer.clear_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
        print(f'Epoch {epoch}  | Loss {avg_loss / float(len(train_loader))}')
        return avg_loss / float(len(train_loader))


def spiltdata(X, ep_per_batch, Nway, Kshot, Qquary):
    """
    把dataloader读出的数据给切分一下，原始的数据格式为(ep_per_batch，Nway，Kshot+Qquary)
    :param X: dataloader读出的X部分，也就是样本部分，由于小样本里label要生成虚拟的，所以不要dataloader读出来的label
    :param ep_per_batch: 几个metabatch
    :param Nway:
    :param Kshot:
    :param Qquary:
    :return: support集的X，query集的X以及对应的label，且query已被随机打乱。
    返回的shape如下所示，*imageshape为图片尺寸，比如3*84*84
    """
    imageshape = X.shape[-3:]
    label = paddle.arange(Nway).unsqueeze(1).tile((1, Qquary)).tile((ep_per_batch, 1)).reshape((ep_per_batch, -1))
    splabel = paddle.arange(Nway).unsqueeze(1).tile((1, Kshot)).tile((ep_per_batch, 1)).reshape((ep_per_batch, -1))

    X = X.reshape((ep_per_batch, Nway, Kshot + Qquary, *imageshape))
    spx = X[:, :, :Kshot].reshape((ep_per_batch, -1, *imageshape))
    qrx = X[:, :, Kshot:].reshape((ep_per_batch, -1, *imageshape))

    return spx, splabel, qrx, label


def get_baseleinpp_encoder():
    return paddle.nn.Sequential(
        pdbackbones.Conv(input_size=(3, 84, 84), output_size=200, conv_channels=[64, 64, 64, 64]).conv,
        paddle.nn.Flatten())


def get_baselinepp_model():
    baselinbackbone = get_baseleinpp_encoder()
    return BaselineTrain(baselinbackbone, 200, loss_type='dist')


def baselinepp_train(base_loader, model, optimizer, start_epoch=0, stop_epoch=400, save_freq=50, savepath=None):
    if savepath == None:
        savepath = os.path.join(os.curdir, 'save')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    train_log = get_log('baselinepp_trainlog', os.path.join(savepath, 'baselinepp_trainlog.log'))

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        trainloss = model.train_loop(epoch, base_loader, optimizer)  # model are called by reference, no need to return
        train_log.log_something(f'Epoch {epoch}  | Loss {trainloss}')
        model.eval()
        if (epoch % save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(savepath, f'baselinepp_{epoch}')
            paddle.save(model.state_dict(), outfile + '_model')
            paddle.save(optimizer.state_dict(), outfile + '_optimzer')
    return model


def baselinepp_test(test_loader, encoder_statedict, numepochs, n_way=5, n_support=1, n_query=16, savepath=None):
    if savepath == None:
        savepath = os.path.join(os.curdir, 'save')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    test_log = get_log('baselinepp_testlog', os.path.join(savepath, 'baselinepp_testlog.log'))

    fsmodel = BaselineFinetune(get_baseleinpp_encoder(), n_way=n_way, n_support=n_support, n_query=n_query,
                               loss_type='dist')
    fsmodel.set_state_dict(encoder_statedict)
    for epoch in range(numepochs):
        acclist = [None] * len(test_loader)
        i=0
        for data, _ in tqdm(test_loader, desc=f'fs{n_support}:', leave=False):
            # 对data进行拆分，必须要做 这个代码不能跑  需要明确需求再修改
            spx, _, qry, label = spiltdata(data, 1, n_way, n_support, n_query)
            feature = fsmodel(data).detach()
            # 把feature给reshape成 nway,(kshot+q_query),1600  的shape
            scores = fsmodel.set_forward_adaptation(feature.reshape((n_way, n_support + n_query, 1600)))
            acclist[i] = paddle.static.accuracy(scores, label.reshape((-1, 1))).numpy()
            i+=1
        acclist = np.array(acclist)
        print(f'epoch{epoch + 1} : fs_{n_support}shot : {acclist.mean():.4f}')
        test_log.log_something(f'epoch{epoch + 1} : fs_{n_support}shot : {acclist.mean():.4f}')
    return fsmodel
