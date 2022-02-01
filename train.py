import torch
from torch import nn
import data
import bert
import _util
import matplotlib.pyplot as plt

# 这里制作最简易的初始化，不封装

batch_size, max_len = 512, 64

# 数据核心
train_iter, vocab = data.load_data_wiki(batch_size, max_len)

# 模型核心
net = bert.BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                    ffn_num_input=128, ffn_num_hiddens=256, num_heads=4,
                    num_layers=4, dropout=0.2, key_size=128, query_size=128,
                    value_size=128, hid_in_features=128, mlm_in_features=128,
                    nsp_in_features=128)
devices = _util.try_gpus()
loss = nn.CrossEntropyLoss()

loss_mlm = []
loss_nsp = []


# 正式训练


def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # 计算遮蔽语言模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 计算下一句子预测任务的损失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l


def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01) 

    # 在下游任务使用 AdamW
    step, timer = 0, _util.Timer()

    # animator = d2l.Animator(xlabel='step', ylabel='loss',xlim=[1, num_steps], legend=['mlm', 'nsp'])

    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = _util.Accumulator(4)

    num_steps_reached = False

    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            print(step)

            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()

            # animator.add(step + 1,(metric[0] / metric[3], metric[1] / metric[3]))
            loss_mlm.append(metric[0]/metric[3])
            loss_nsp.append(metric[1]/metric[3])

            print(f'MLM loss {metric[0] / metric[3]:.3f}, '
                f'NSP loss {metric[1] / metric[3]:.3f}')
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
    print(len(loss_mlm))
    torch.save(net, 'myBert')
    torch.save(net.state_dict(), 'myBert.pth')

def loss_graph(file_name):
    n = len(loss_mlm)
    plt.title('pretrain BERT')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.plot(range(0,n),loss_mlm, label = 'mlm')
    plt.plot(range(0,n),loss_nsp, label = 'nsp')
    plt.legend()
    plt.savefig(file_name)


if __name__ == '__main__':

    train_bert(train_iter, net, loss, len(vocab), devices, 50)

    loss_graph('pretrain_loss.jpg')


