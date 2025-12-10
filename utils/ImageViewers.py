from ipywidgets import interact, interactive
from ipywidgets import widgets
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os



def myshow_comp(img_1, img_2, title=None, margin=0.05, dpi=80, cmap="gray", fig_size_multiplier=1.0):
    """ 
    Deprecated function, included for backward compatibility, use myshow_composition instead.
    Display two images side by side for comparison. 
    If the images are 3D, a slider is added to scroll through the z-axis. 
    """
    nda_1 = sitk.GetArrayFromImage(img_1)
    nda_2 = sitk.GetArrayFromImage(img_2)

    spacing = img_1.GetSpacing()
    slicer = False

    if nda_1.ndim == 3:
        # fastest dim, either component or x
        c = nda_1.shape[-1]

        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3, 4):
            slicer = True

    elif nda_1.ndim == 4:
        c = nda_1.shape[-1]

        if not c in (3, 4):
            raise RuntimeError("Unable to show 3D-vector Image")

        # take a z-slice
        slicer = True

    if slicer:
        ysize = nda_1.shape[1]
        xsize = nda_1.shape[2]
    else:
        ysize = nda_1.shape[0]
        xsize = nda_1.shape[1]

    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = ((1 + margin) * ysize / dpi)*fig_size_multiplier, ((1 + margin) * xsize / dpi)*fig_size_multiplier

    def callback(z=None):
        extent = (0, xsize * spacing[1], ysize * spacing[0], 0)

        fig = plt.figure(figsize=figsize, dpi=dpi)

        # Make the axis the right size...
        ax = fig.add_axes([margin, margin, (1 - 2 * margin)*0.5, 1 - 2 * margin])
        ax_2 = fig.add_axes([2*margin+0.5, margin, (1 - 2 * margin)*0.5, 1 - 2 * margin])

        if z is None:
            ax.imshow(nda_1, extent=extent, interpolation=None, cmap=cmap)   #### Add here aspect ratio derived from the img direction array
            ax_2.imshow(nda_2, extent=extent, interpolation=None, cmap=cmap)
        else:
            ax.imshow(nda_1[z, ...], extent=extent, interpolation=None, cmap=cmap)
            ax_2.imshow(nda_2[z, ...], extent=extent, interpolation=None, cmap=cmap)

        if title:
            plt.title(title)

        plt.show()

    if slicer:
        interact(callback, z=(0, nda_1.shape[0] - 1))
    else:
        callback()



def myshow_selector(img_dir, always_shown=None, **kwargs):
    """
    Wrapper for myshow_composition that shows a dropdown 
    Args:
        img_dir (list): directory containing images.
        always_shown (list): list of image file names that are always shown in addition to the selected image.
        **kwargs: Extra arguments passed to myshow_composition.
    """
    img_path_list = [os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith(".nii.gz")]
    ### remove and store seperately always shown images
    if always_shown is not None:
        always_shown_paths = [os.path.join(img_dir, img) for img in always_shown if img in os.listdir(img_dir)]
        img_path_list = [img for img in img_path_list if os.path.basename(img) not in always_shown]

    def generate_titles(img_path_list):
        return [os.path.basename(img).replace(".nii.gz", "").replace("slice_aligned_", "") for img in img_path_list]


    titles = generate_titles(img_path_list)

    ### load images
    img_list = [sitk.ReadImage(img_path) for img_path in img_path_list]
    if always_shown is not None:
        always_shown_imgs = [sitk.ReadImage(img_path) for img_path in always_shown_paths]
        always_shown_titles = generate_titles(always_shown_paths)

    dropdown = widgets.Dropdown(
        options=list(zip(titles, range(len(img_list)))),
        description="Select image:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )

    def _update(idx):
        myshow_composition([img_list[idx]]+always_shown_imgs, title=[titles[idx]]+always_shown_titles, **kwargs)

    widgets.interact(_update, idx=dropdown)


def myshow_composition(img_list, title=None, margin=0.05, dpi=80, cmap="gray", fig_size_multiplier=1.0):
    nda_list= [sitk.GetArrayFromImage(img) for img in img_list]
    spacing = img_list[0].GetSpacing()
    nr_images = len(img_list)
    slicer = False
    channel_list=[]
    for i in range(nr_images):
        c=1
        if nda_list[i].ndim == 3:
            # fastest dim, either component or x
            c = nda_list[i].shape[-1]

            # the the number of components is 3 or 4 consider it an RGB image
            if not c in (3, 4):
                slicer = True
                c=1
        elif nda_list[i].ndim == 4:
            c = nda_list[i].shape[-1]

            if not c in (3, 4):
                raise RuntimeError("Unable to show 3D-vector Image")

            # take a z-slice
            slicer = True
        channel_list.append(c)
        
    if slicer:
        ysize = nda_list[0].shape[1]
        xsize = nda_list[0].shape[2]
    else:
        ysize = nda_list[0].shape[0]
        xsize = nda_list[0].shape[1]

    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = ((1 + margin) * ysize / dpi)*fig_size_multiplier, ((1 + margin) * xsize / dpi)*fig_size_multiplier

    def callback(z=None):
        extent = (0, xsize * spacing[1], ysize * spacing[0], 0)

        fig = plt.figure(figsize=figsize, dpi=dpi)

        # Make the axis the right size...
        axes=[]
        for i in range(nr_images):
            if i == 0:
                ax = fig.add_axes([margin, margin, (1 - nr_images * margin)*(1/nr_images), 1 - 2 * margin])
            else:
                ax = fig.add_axes([margin*(i+1)+(1 - nr_images * margin)*(1/nr_images)*i, margin, (1 - nr_images * margin)*(1/nr_images), 1 - 2 * margin])
            axes.append(ax)

        if z is None:
            for i, ax in enumerate(axes):
                ax.imshow(nda_list[i], extent=extent, interpolation=None, cmap=cmap)
        else:
            for i, ax in enumerate(axes):
                ax.imshow(nda_list[i][z, ...], extent=extent, interpolation=None, cmap=cmap)

        if title:
            plt.title(title)

        plt.show()

    if slicer:
        interact(callback, z=(0, nda_list[0].shape[0] - 1))
    else:
        callback()

def myshow(img, title=None, margin=0.05, dpi=80, cmap="gray", fig_size_multiplier=1.0):
    nda = sitk.GetArrayFromImage(img)

    spacing = img.GetSpacing()
    slicer = False

    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]

        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3, 4):
            slicer = True

    elif nda.ndim == 4:
        c = nda.shape[-1]

        if not c in (3, 4):
            raise RuntimeError("Unable to show 3D-vector Image")

        # take a z-slice
        slicer = True

    if slicer:
        ysize = nda.shape[1]
        xsize = nda.shape[2]
    else:
        ysize = nda.shape[0]
        xsize = nda.shape[1]

    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = ((1 + margin) * ysize / dpi)*fig_size_multiplier, ((1 + margin) * xsize / dpi)*fig_size_multiplier

    def callback(z=None):
        extent = (0, xsize * spacing[1], ysize * spacing[0], 0)

        fig = plt.figure(figsize=figsize, dpi=dpi)

        # Make the axis the right size...
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

        if z is None:
            ax.imshow(nda, extent=extent, interpolation=None, cmap=cmap)   #### Add here aspect ratio derived from the img direction array
        else:
            ax.imshow(nda[z, ...], extent=extent, interpolation=None, cmap=cmap)

        if title:
            plt.title(title)

        plt.show()

    if slicer:
        interact(callback, z=(0, nda.shape[0] - 1))
    else:
        callback()


def myshow3d(img, xslices=[], yslices=[], zslices=[], title=None, margin=0.05, dpi=80):
    size = img.GetSize()
    img_xslices = [img[s, :, :] for s in xslices]
    img_yslices = [img[:, s, :] for s in yslices]
    img_zslices = [img[:, :, s] for s in zslices]

    maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))

    img_null = sitk.Image([0, 0], img.GetPixelID(), img.GetNumberOfComponentsPerPixel())

    img_slices = []
    d = 0

    if len(img_xslices):
        img_slices += img_xslices + [img_null] * (maxlen - len(img_xslices))
        d += 1

    if len(img_yslices):
        img_slices += img_yslices + [img_null] * (maxlen - len(img_yslices))
        d += 1

    if len(img_zslices):
        img_slices += img_zslices + [img_null] * (maxlen - len(img_zslices))
        d += 1

    if maxlen != 0:
        if img.GetNumberOfComponentsPerPixel() == 1:
            img = sitk.Tile(img_slices, [maxlen, d])
        # TODO check in code to get Tile Filter working with VectorImages
        else:
            img_comps = []
            for i in range(0, img.GetNumberOfComponentsPerPixel()):
                img_slices_c = [sitk.VectorIndexSelectionCast(s, i) for s in img_slices]
                img_comps.append(sitk.Tile(img_slices_c, [maxlen, d]))
            img = sitk.Compose(img_comps)

    myshow(img, title, margin, dpi)



def checkering(img_1, img_2, tile_size=25):
    """
    Create a checkerboard image of the two input images. The optional tile_size parameter gouverns the side length of the checkerboard tiles
    Checkering is done over the first two dimensions of the input images, ignoring potential trailing dimensions and color channels
    """
    assert (img_1.shape==img_2.shape)
    img_c=np.zeros(img_1.shape)
    tiles=[int(img_1.shape[-2]/tile_size), int(img_1.shape[-1]/tile_size)]
    for i in range(tiles[-2]):
        for j in range(tiles[-1]):
            if((i+j)%2==0):
                img_c[..., i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]=img_1[..., i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
            else:
                img_c[..., i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]=img_2[...,i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]

    return img_c

def checkering_color(img_1, img_2, tile_size=25):
    """
    Create a checkerboard image of the two input images. The optional tile_size parameter gouverns the side length of the checkerboard tiles
    Checkering is done over the first two dimensions of the input images, ignoring potential trailing dimensions and color channels
    """
    assert (img_1.shape==img_2.shape)
    img_c=np.zeros(img_1.shape)
    tiles=[int(img_1.shape[0]/tile_size), int(img_1.shape[1]/tile_size)]
    for i in range(tiles[-2]):
        for j in range(tiles[-1]):
            if((i+j)%2==0):
                img_c[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]=img_1[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
            else:
                img_c[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]=img_2[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]

    return img_c

def myshow_overlay(img_1, img_2, title=None, margin=0.05, dpi=80, cmap="gray", fig_size_multiplier=1.0):
    nda_1 = sitk.GetArrayFromImage(img_1)
    nda_2 = sitk.GetArrayFromImage(img_2)

    spacing = img_1.GetSpacing()
    slicer = False

    if nda_1.ndim == 3:
        # fastest dim, either component or x
        c = nda_1.shape[-1]

        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3, 4):
            slicer = True

    elif nda_1.ndim == 4:
        c = nda_1.shape[-1]

        if not c in (3, 4):
            raise RuntimeError("Unable to show 3D-vector Image")

        # take a z-slice
        slicer = True

    if slicer:
        ysize = nda_1.shape[1]
        xsize = nda_1.shape[2]
    else:
        ysize = nda_1.shape[0]
        xsize = nda_1.shape[1]

    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = ((1 + margin) * ysize / dpi)*fig_size_multiplier, ((1 + margin) * xsize / dpi)*fig_size_multiplier

    def callback(z=None):
        extent = (0, xsize * spacing[1], ysize * spacing[0], 0)

        fig = plt.figure(figsize=figsize, dpi=dpi)

        # Make the axis the right size...
        ax = fig.add_axes([margin, margin, (1 - 2 * margin), 1 - 2 * margin])

        if z is None:
            ax.imshow(nda_1, extent=extent, interpolation=None, cmap=cmap, alpha=0.5)   #### Add here aspect ratio derived from the img direction array
            ax.imshow(nda_2, extent=extent, interpolation=None, cmap=cmap, alpha=0.5)
        else:
            ax.imshow(nda_1[z, ...], extent=extent, interpolation=None, cmap=cmap, alpha=0.5)
            ax.imshow(nda_2[z, ...], extent=extent, interpolation=None, cmap=cmap, alpha=0.5)

        if title:
            plt.title(title)

        plt.show()

    if slicer:
        interact(callback, z=(0, nda_1.shape[0] - 1))
    else:
        callback()



def checkering_sitk(sitk_img_1: sitk.Image, sitk_img_2: sitk.Image, tile_size=25):
    """
    Create a checkerboard image of the two input sitk images. The optional tile_size parameter gouverns the side length of the checkerboard tiles
    Checkering is done over the first two dimensions of the input images, ignoring potential trailing dimensions and color channels
    ATTENTION!: Does not check if all meta data match, so returned sitk image just copies meta info from sitk_img_1
    """
    assert (sitk_img_1.GetSize()==sitk_img_2.GetSize())
    img_1=sitk.GetArrayFromImage(sitk_img_1)
    img_2=sitk.GetArrayFromImage(sitk_img_2)
    isVector=(sitk_img_1.GetNumberOfComponentsPerPixel()>1)
    if (isVector):
        img_c=checkering_color(img_1, img_2, tile_size)
        img_c=img_c.astype(img_1.dtype)
    else:
        img_c=checkering(img_1, img_2, tile_size)
    img_c_sitk=sitk.GetImageFromArray(img_c, isVector=isVector)
    img_c_sitk.CopyInformation(sitk_img_1)
    return img_c_sitk


def show_grid(images, names, n_images=15):
    fig = plt.figure(figsize=(300., 150.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(int(np.ceil(n_images/5)), min(5, len(images))),  # creates 1x2 grid of axes
                 axes_pad=0.5,  # pad between axes in inch.
                 )

    for ax, im, name in zip(grid, images, names):
    # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap='gray')
        ax.set_title(name)

    plt.show()