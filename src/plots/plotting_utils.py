import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator


def format_axes(ax: Axes | list[Axes], **kwargs):
    # This handles if two axes are passed in, there is some default styling always done to these
    if isinstance(ax, list) and len(ax) == 2:
        format_axes(
            ax[0], ticks_right=False, **(kwargs[0] if 0 in kwargs.keys() else {})
        )
        format_axes(
            ax[1], ticks_left=False, **(kwargs[1] if 1 in kwargs.keys() else {})
        )

        if 'combine_legends' in kwargs.keys() and kwargs['combine_legends'] is True:
            handles, labels = ax[0].get_legend_handles_labels()
            handles2, labels2 = ax[1].get_legend_handles_labels()

            # Combine the handles and labels
            handles.extend(handles2)
            labels.extend(labels2)

            # into  a single legend
            ax[0].legend(handles, labels, loc='best' if 'legend_loc' not in kwargs.keys() else kwargs['legend_loc'])

        return

    if ax.get_legend():
        ax.legend(
            facecolor='white',
            loc='best' if 'legend_loc' not in kwargs.keys() else kwargs['legend_loc'],
        )

    # Make the axes the plots have a white background
    ax.set_facecolor('white')

    # Format the spines
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_edgecolor('k')
        ax.spines[side].set_linewidth(0.5)

    # Add minor ticks to the axes
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Turn on all ticks
    ax.tick_params(
        which='both',
        top=True if 'ticks_top' not in kwargs.keys() else kwargs['ticks_top'],
        bottom=True if 'ticks_bottom' not in kwargs.keys() else kwargs['ticks_bottom'],
        left=True if 'ticks_left' not in kwargs.keys() else kwargs['ticks_left'],
        right=True if 'ticks_right' not in kwargs.keys() else kwargs['ticks_right'],
    )

    ax.tick_params(which='minor', length=2, color='k', direction='out')
    ax.tick_params(which='major', length=4, color='k', direction='out')

    if 'autoscale_x' in kwargs.keys() and kwargs['autoscale_x'] is True:
        ax.autoscale(enable=True, tight=True, axis='x')
