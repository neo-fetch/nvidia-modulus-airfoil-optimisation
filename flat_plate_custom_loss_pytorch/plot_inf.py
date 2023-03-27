import matplotlib.pyplot as plt
import numpy as np
import scipy


class _Plotter:
    def __call__(self, *args):
        raise NotImplementedError

    def _add_figures(self, group, name, results_dir, writer, step, *args):
        "Try to make plots and write them to tensorboard summary"

        # catch exceptions on (possibly user-defined) __call__
        try:
            fs = self(*args)
        except Exception as e:
            print(f"error: {self}.__call__ raised an exception:", str(e))
        else:
            for f, tag in fs:
                f.savefig(
                    results_dir + name + "_" + tag + ".png",
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
                writer.add_figure(group + "/" + name + "/" + tag, f, step, close=True)
            plt.close("all")

    def _interpolate_2D(self, size, invar, *outvars):
        "Interpolate 2D outvar solutions onto a regular mesh"

        assert len(invar) == 2

        # define regular mesh to interpolate onto
        xs = [invar[k][:, 0] for k in invar]
        extent = (xs[0].min(), xs[0].max(), xs[1].min(), xs[1].max())
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], size),
            np.linspace(extent[2], extent[3], size),
            indexing="ij",
        )

        # interpolate outvars onto mesh
        outvars_interp = []
        for outvar in outvars:
            outvar_interp = {}
            for k in outvar:
                outvar_interp[k] = scipy.interpolate.griddata(
                    (xs[0], xs[1]), outvar[k][:, 0], tuple(xyi)
                )
            outvars_interp.append(outvar_interp)

        return [extent] + outvars_interp


class PlotInference(_Plotter):
    "Default plotter class for inferencer"

    def __call__(self, invar, outvar):
        "Default function for plotting inferencer data"
        invar = {"x": invar["x"], "y": invar["y"]}

        ndim = len(invar)
        if ndim > 2:
            print("Default plotter can only handle <=2 input dimensions, passing")
            return []

        # interpolate 2D data onto grid
        if ndim == 2:
            extent, outvar = self._interpolate_2D(100, invar, outvar)

        # make plots
        dims = list(invar.keys())
        fs = []
        for k in outvar:
            f = plt.figure(figsize=(5, 4), dpi=100)
            if ndim == 1:
                plt.plot(invar[dims[0]][:, 0], outvar[:, 0])
                plt.xlabel(dims[0])
            elif ndim == 2:
                plt.imshow(outvar[k].T, origin="lower", extent=extent)
                plt.xlabel(dims[0])
                plt.ylabel(dims[1])
                plt.colorbar()
            plt.title(k)
            plt.tight_layout()
            fs.append((f, k))

        return fs
