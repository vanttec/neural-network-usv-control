import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

class Graph3D():
    '''Displays a 3D graph'''
    def __init__(self, xlim=None, ylim=None, zlim=None):
        '''Initialize graph parameters
        Params
        ======
            xlim = float list, x-axis limits
            ylim = float list, y-axis limits
            zlim = float list, z-axis limits
        '''
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.fig = plt.figure()
        self.reset()

    def reset(self):
        '''Reset the graph'''
        self.ax = p3.Axes3D(self.fig)
        if self.xlim != None: self.ax.set_xlim(self.xlim[0], self.xlim[1])
        if self.ylim != None: self.ax.set_ylim(self.ylim[0], self.ylim[1])
        if self.zlim != None: self.ax.set_zlim(self.zlim[0], self.zlim[1])
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

    def set_lims(self, xlim=None, ylim=None, zlim=None):
        '''Set new axes limits
        Params
        ======
            xlim = float list, x-axis limits
            ylim = float list, y-axis limits
            zlim = float list, z-axis limits
        '''
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        if self.xlim != None: self.ax.set_xlim(self.xlim[0], self.xlim[1])
        if self.ylim != None: self.ax.set_ylim(self.ylim[0], self.ylim[1])
        if self.zlim != None: self.ax.set_zlim(self.zlim[0], self.zlim[1])

    def _update_lines(self, num, data_lines, props, lines):
        '''Function used to updates the animation'''
        if props is None: props = [['o','black',5] for _ in data_lines]
        for line, prop, data in zip(lines, props, data_lines):
            if prop[0] != 'o-o':
                line.set_data(data[0:2, num])
                line.set_3d_properties(data[2, num])
            if prop[0] == 'o':
                line.set_marker('o')
                line.set_markersize(prop[2])
                line.set_linestyle('')
            elif prop[0] == '-':
                line.set_linewidth(prop[2])
            elif prop[0] == 'o-':
                line.set_marker('o')
                line.set_markersize(prop[2])
                line.set_linewidth(prop[3])
            elif prop[0] == 'o-o':
                line.set_data(data[0:2, :num])
                line.set_3d_properties(data[2, :num])
                line.set_marker('o')
                line.set_markersize(prop[2])
                line.set_linewidth(prop[3])
            line.set_color(prop[1])
        return lines

    def show(self, data_lines=None, props=None, interval=30, repeat=False):
        '''Use data lines and their properties to display them
        Params
        ======
            data_lines: list, points in the form [obj1,obj2...]
                obj1: numpy array in the form [x1,y1,z1]
                x1: 1D numpy array for points and
                    2D numpy array for lines
            props: list, properties in the form [[mrk,color,size1,size2],...]
                mrkr: 'o', '-', 'o-', 'o-o'(continous display)
                color: any matplotlib color
                size1: size of marker 1
                seze2: size of marker 2 if present
            interval: int, interval between frames in miliseconds
            repeat: bool, repeat animation
        '''
        if data_lines != None:
            lines = [self.ax.plot([], [], [])[0] for data in data_lines]
            lines_ani = animation.FuncAnimation(
                self.fig,
                self._update_lines,
                len(data_lines[0][0]),
                fargs=(data_lines, props, lines),
                interval=interval,
                repeat=repeat)
        plt.pause(1e-10)
