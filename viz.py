import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss_metrics(path, history):
    fig = plt.figure()
    plt.plot(history['train']['iter'], history['train']['err'],
             color='b', label='training')
    plt.plot(history['train_ao']['iter'], history['train_ao']['err'],
             color='r', label='train_ao')
    plt.plot(history['train_av']['iter'], history['train_av']['err'],
             color='g', label='train_av')
    plt.plot(history['val_av']['iter'], history['val_av']['err'],
             color='c', label='val_av')
    plt.plot(history['val_ao']['iter'], history['val_ao']['err'],
             color='y', label='val_ao')

    plt.legend()
    fig.savefig(os.path.join(path, 'loss.png'), dpi=200)
    plt.close('all')

    if history['val_av']['iter'] != []:
        fig = plt.figure()
        plt.plot(history['val_av']['iter'], history['val_av']['sdr'],
                color='r', label='SDR')
        plt.plot(history['val_av']['iter'], history['val_av']['sir'],
                color='g', label='SIR')
        plt.plot(history['val_av']['iter'], history['val_av']['si_sdr'],
                color='b', label='SI_SDR')
        plt.legend()
        fig.savefig(os.path.join(path, 'metrics_av.png'), dpi=200)
        plt.close('all')

    if history['val_ao']['iter'] != []:
        fig = plt.figure()
        plt.plot(history['val_ao']['iter'], history['val_ao']['sdr'],
                color='r', label='SDR')
        plt.plot(history['val_ao']['iter'], history['val_ao']['sir'],
                color='g', label='SIR')
        plt.plot(history['val_ao']['iter'], history['val_ao']['si_sdr'],
                color='b', label='SI_SDR')
        plt.legend()
        fig.savefig(os.path.join(path, 'metrics_ao.png'), dpi=200)
        plt.close('all')


class HTMLVisualizer():
    def __init__(self, fn_html):
        self.fn_html = fn_html
        self.content = '<table>'
        self.content += '<style> table, th, td {border: 1px solid black;} </style>'

    def add_header(self, elements):
        self.content += '<tr>'
        for element in elements:
            self.content += '<th>{}</th>'.format(element)
        self.content += '</tr>'

    def add_rows(self, rows):
        for row in rows:
            self.add_row(row)

    def add_row(self, elements):
        self.content += '<tr>'

        # a list of cells
        for element in elements:
            self.content += '<td>'

            # fill a cell
            for key, val in element.items():
                if key == 'text':
                    self.content += val
                elif key == 'image':
                    self.content += '<img src="{}" style="max-height:256px;max-width:256px;">'.format(val)
                elif key == 'audio':
                    self.content += '<audio controls><source src="{}"></audio>'.format(val)
                elif key == 'video':
                    self.content += '<video src="{}" controls="controls" style="max-height:256px;max-width:256px;">'.format(val)
            self.content += '</td>'

        self.content += '</tr>'

    def write_html(self):
        self.content += '</table>'
        with open(self.fn_html, 'w') as f:
            f.write(self.content)
