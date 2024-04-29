import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec


class PdfPlotter(object):
    def __init__(self, filepath, closefigs=True, fixed_margins=None):
        if closefigs:
            for fignum in plt.get_fignums():
                plt.close(fignum)

        self.plot_number = 0
        self.filepath = filepath
        self.figs = []
        if '_pdf_plotter' not in plt.__dict__.keys():
            plt._figure = plt.figure
            plt._show = plt.show
        plt.figure = self.figure
        plt.show = self.save
        plt._pdf_plotter = self

        self._fixed_margins = fixed_margins


    def save(self):
        pp = PdfPages(self.filepath)
        for figure in self.figs:
            fig = plt._figure(figure.number)
            if self._fixed_margins is not None:
                figsize = fig.get_size_inches()*fig.dpi
                margins = dict(left=self._fixed_margins['left']/figsize[0],
                    right=(figsize[0] - self._fixed_margins['right'])/figsize[0],
                    bottom=self._fixed_margins['bottom']/figsize[1],
                    top=(figsize[1] - self._fixed_margins['top'])/figsize[1])
                plt.subplots_adjust(**margins)
            plt.savefig(pp, format='pdf')

        pp.close()


    def figure(self, num=None, **kwargs):
        if num is not None:
            return plt._figure(num=num, **kwargs)

        fig = plt._figure(**kwargs)
        self.figs.append(fig)
        self.plot_number = 0

        return fig


class AxisIterator:
    def __init__(self, gs_args, fig_args):
        plt.figure(**fig_args)
        self._gs = gridspec.GridSpec(**gs_args)
        self._fig_args = fig_args
        self._count = iter(range(self._gs._nrows * self._gs._ncols))
        self._axs = []

    def next(self):
        try:
            ax = plt.subplot(self._gs[next(self._count)])
            self._axs.append(ax)
            return ax
        except StopIteration:
            plt.figure(**self._fig_args)
            self._count = iter(range(self._gs._nrows * self._gs._ncols))
            self._axs = []
            return self.next()

    @property
    def nrows(self):
        return self._gs._nrows

    @property
    def ncols(self):
        return self._gs._ncols

    def __getitem__(self, i):
        return self._axs[i]

    def __len__(self):
        return len(self._axs)

    def to_list(self):
        self._axs.extend(list(map(lambda x: plt.subplot(self._gs[x]), self._count)))
        return tuple(self._axs)
