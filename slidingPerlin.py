
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from utils import set_params, make_animation
import time, os
from rich import print
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn
from rich.console import Console
from rich.table import Table
from datetime import datetime
seed = np.random.seed(1)

import argparse

def GetArgs():

    def intlist(arg): return list(map(int, arg.split(',')))
    
    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = 'Author : Pawlicki Loïc\n' + '─'*30 + '\n'
    )
    parser.add_argument('-s', '--shape',    default = [250, 250],   type = intlist, metavar = '', action= 'store',  help = 'x and y-axis lengthes for meshgrid plane (2 ints)')
    parser.add_argument('-o', '--noct',     default = [1, 1],       type = intlist, metavar = '', action= 'store',  help = 'x and y-axis octaves count for noise generation methods (2 ints)')
    parser.add_argument('-g', '--gamma',    default = 1,            type = float,   metavar = '', action= 'store',  help = 'Frequency octaves amplitude factoring coefficient')
    parser.add_argument('-ds', '--dstep',   default = 0.2,          type = float,   metavar = '', action= 'store',  help = 'Data phase step resolution')
    parser.add_argument('-p', '--path',     default = 'Resources',  type = str,     metavar = '', action= 'store',  help = 'dir path ref for animation output files location')
    parser.add_argument('-f', '--fname',    default = 'auto',       type = str,     metavar = '', action= 'store',  help = 'save fname for animation output .gif file')
    parser.add_argument('-m', '--cmap',     default = 'magma',      type = str,     metavar = '', action= 'store',  help = 'Cmap to be used for output generation')
    parser.add_argument('-st', '--stride',  default = 3,            type = int,     metavar = '', action= 'store',  help = 'Anim cstride & rstride adjust flag')
    parser.add_argument('-fr', '--frate',   default = 30,           type = int,     metavar = '', action= 'store',  help = 'Framerate setpoint')
    parser.add_argument('-nf', '--nframe',  default = 60,           type = int,     metavar = '', action= 'store',  help = 'Number of frames to compute')
    parser.add_argument('-d', '--display',  default = 0,            type = bool,    metavar = '', action= 'store',  help = 'Show output frames in matplotlib')
    parser.add_argument('-no', '--n_files', default = 1,            type = int,     metavar = '', action= 'store',  help = 'Output gif amount using custom ')
    
    args = parser.parse_args()
    argtable = Table(title = 'Parsed arguments ref. table', title_justify = 'left', padding = (0,2))
    argtable.add_column(header = 'Argument', style = 'italic', justify='center')
    argtable.add_column(header = 'Default value', style = 'yellow', justify='left')
    argtable.add_column(header = 'Dtype', style = 'blue', justify='center')
    for arg in vars(args): argtable.add_row(f'--{arg}', f'{getattr(args, arg)}', f'{getattr(args, arg).__class__.__name__}')
    print(argtable)
    return args, argtable

args, argtable = GetArgs()

def format_outfile_name(args):
    return f'py {os.path.basename(__file__)} -s {args.shape[0]},{args.shape[1]} -o {args.noct[0]},{args.noct[1]} -g {args.gamma} -st {args.stride} -ds {args.dstep} -fr {args.frate} -nf {args.nframe} {str(datetime.now())[:-7]}'

if args.fname == 'auto':
    outfile_name = format_outfile_name(args)
else:
    outfile_name = args.fname


# Meshgrid plane shape
n_x, n_y = args.shape[-2], args.shape[-1]
# Number of octaves in each direction.
ncx, ncy = args.noct[-2], args.noct[-1]

xprms, yprms = set_params(ncx, ncy, args.gamma)
x, y = np.meshgrid(np.arange(n_x), np.arange(n_y))

def plot_wireframe(ax, X, Y, Z):
    
    # plot Z surface as a surface (wireframe-like)
    surf = ax.plot_surface(X, Y, Z, rstride=args.stride, cstride=args.stride, cmap=args.cmap)
    ax.set_xlim(0, n_x)
    ax.set_ylim(0, n_y)
    ax.axis('off')
    
    # custom viewpoint for .gif rendering
    # ax.elev = 45
    ax.elev = 42
    ax.azim = 35
    
    # force aspect ratio on each axis
    ax.set_box_aspect((n_x, n_y, np.ptp(Z)*100)) 
    return surf

def update_prms(xprms, yprms, step):
    xprms[2] += step * (np.random.random(xprms[2].shape) - 1)
    yprms[2] += step * (np.random.random(yprms[2].shape) - 1)
    # ! print(f'{xprms=}{yprms=}')
    return xprms, yprms

def make_progress(console, estim: bool = False,):
    columns = [
        SpinnerColumn(speed = 1), 
        TextColumn(
            text_format = '{task.description}',
            style = 'blue'
        ),
        BarColumn(), 
        TimeElapsedColumn(),
        TaskProgressColumn()
        ]
    if estim: columns.append(TimeRemainingColumn(elapsed_when_finished=True))
    progress = Progress(*columns, console = console)
    return progress


console = Console()
progress = make_progress(console, estim = True)

task = progress.add_task(description = 'Anim computation', total = None)
t_start = time.time()

with progress:
    ani = make_animation(
        x = x, 
        y = y, 
        xprms = xprms, 
        yprms = yprms, 
        step = args.dstep,
        update = update_prms,
        plot = plot_wireframe,
        console = console,
        progress = progress,
        save = True, 
        nframes = args.nframe, 
        fps = args.frate,
        filename = os.path.join(args.path, outfile_name)
    )
    progress.update(task, completed = True)
    progress.stop()
    
print(f'Computed {n_x*n_y*args.nframe:.2e} points, done in {time.time()-t_start:.2f} seconds, data saved at {os.path.join(args.path, args.fname)}.gif')
print(argtable)
if args.display:
    time.sleep(1)
    plt.show()
    
    