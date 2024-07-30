import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import argparse

DPI = 100
VMIN = -1000
VMAX = 170


def save_ct(data, spacing, filename):

    sizes = np.shape(data)
    xsize = sizes[1]
    ysize = sizes[0]
    xspacing = spacing[1]
    yspacing = spacing[0]
    fig = plt.figure(
        figsize=((xspacing * xsize) / float(DPI), (yspacing * ysize) / float(DPI)),
        dpi=DPI,
    )
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    extent = (0, xsize * xspacing, ysize * yspacing, 0)
    ax.imshow(data, cmap="gray", extent=extent, interpolation="bicubic")
    print(f"save_ct {filename}")
    plt.savefig(filename, dpi=sizes[0])
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctfn", type=str)
    parser.add_argument("--maskfn", type=str)
    parser.add_argument("--outfn", type=str)
    args = parser.parse_args()
    ctfn = args.ctfn
    maskfn = args.maskfn
    outfn = args.outfn

    im = sitk.ReadImage(ctfn)
    mask = sitk.ReadImage(maskfn)

    im = sitk.Cast(
        sitk.IntensityWindowing(
            im,
            windowMinimum=VMIN,
            windowMaximum=VMAX,
            outputMinimum=0.0,
            outputMaximum=255.0,
        ),
        sitk.sitkUInt8,
    )

    cyan = [0, 255, 255]
    purple = [128, 0, 128]
    contour_image = sitk.LabelMapContourOverlay(
        sitk.Cast(mask, sitk.sitkLabelUInt8),
        im,
        opacity=1,
        sliceDimension=0,
        contourThickness=[3, 3, 0],
        dilationRadius=[3, 3, 0],
        colormap=purple + cyan,
    )

    overlay_image = sitk.LabelOverlay(
        image=im,
        labelImage=mask,
        opacity=0.5,
        backgroundValue=0,
        colormap=purple + cyan,
    )
    imnp = sitk.GetArrayFromImage(im)
    contournp = sitk.GetArrayFromImage(contour_image)
    overlaynp = sitk.GetArrayFromImage(overlay_image)

    spacing = np.array(im.GetSpacing())
    print(spacing)

    masknp = sitk.GetArrayFromImage(mask)
    inds = np.where(masknp > 0)
    zmin = np.min(inds[0]) + 5
    zmax = np.max(inds[0]) - 5
    num_slices = 20
    zslices = np.linspace(zmin, zmax, num_slices)
    zslices = np.round(zslices).astype(np.int16)

    for slc in zslices:
        save_ct(imnp[slc, :, :], spacing, f"{outfn}_ct_{slc}.pdf")
        save_ct(contournp[slc, :, :], spacing, f"{outfn}_contour_{slc}.pdf")
        save_ct(overlaynp[slc, :, :], spacing, f"{outfn}_overlay_{slc}.pdf")


if __name__ == "__main__":
    main()
