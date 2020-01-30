import os
import numpy
import torch

from matplotlib import pyplot


def calc_gaussian_mixture(tanh_layer, params, bias=0.):
    # mixture of guassian computation, in the paper, fomular(15) ~ (22)
    mdn_params = params.narrow(-1, 0, params.size()[-1] - 1)
    pi_hat, mu1, mu2, sigma1_hat, sigma2_hat, rho_hat \
        = mdn_params.chunk(6, dim=-1)
    end = torch.sigmoid(params.narrow(-1, params.size()[-1] - 1, 1))
    rho = tanh_layer(rho_hat)
    weights = torch.softmax(pi_hat * (1 + bias), dim=-1) # no bias during training
    # sigma1, sigma2 = torch.exp(sigma1_hat), torch.exp(sigma2_hat)

    # adaptive to formula (21) to (61) and (62)
    sigma1, sigma2 = torch.exp(sigma1_hat - bias), torch.exp(sigma2_hat - bias)
    return end, weights, mu1, mu2, sigma1, sigma2, rho


def save_loss_figure(t_loss, v_loss):
    f1 = plt.figure(1)
    if t_loss:
        plt.plot(range(1, args.num_epochs + 1),
                t_loss,
                color='blue',
                linestyle='solid')
    if v_loss:
        plt.plot(range(1, args.num_epochs + 1),
                v_loss,
                color='red',
                linestyle='solid')
    f1.savefig(args.task + "_loss_curves", bbox_inches='tight')


def save_checkpoint(epoch, model, v_loss, optimizer,
                    directory, filename='best.pt'):
    checkpoint = ({
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'v_loss': v_loss,
        'optimizer': optimizer.state_dict()
    })
    try:
        torch.save(checkpoint, os.path.join(directory, filename))
    except:
        os.mkdir(directory)
        torch.save(checkpoint, os.path.join(directory, filename))


def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    x = numpy.cumsum(stroke[:, 1])
    y = numpy.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = numpy.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    pyplot.close()
