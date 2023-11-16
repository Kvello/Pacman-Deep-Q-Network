import matplotlib.pyplot as plt
import numpy as np
import pickle
import click

@click.command()
@click.option("--file", default=None, help='File to plot')
@click.option('--title', default='Plot', help='Title of the plot')
@click.option('--save', is_flag=True, help='Save the plot')
@click.option('--show', is_flag=True, help='Show the plot')
@click.option('--format', default='pdf', help='Format of the saved plot')
@click.option('--xlabel', default='x', help='Label of the x-axis')
@click.option('--ylabel', default='y', help='Label of the y-axis')
def main(title, file, save, show, format, xlabel, ylabel):
    if file is None:
        raise ValueError('No file specified')
    if file.endswith('.npy'):
        data = np.load(file)
    elif file.endswith('.pkl'):
        with open(file, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError('File type not supported')
    plt.plot(data)
    plt.title(title)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save:
        plt.savefig(f'{title}.{format}')
    if show:
        plt.show()
    
if __name__ == '__main__':
    main()