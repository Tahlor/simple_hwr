from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from math import ceil
from PIL import Image, ImageDraw
import os
import cv2
from hwr_utils.stroke_recovery import *
from hwr_utils import utils, stroke_recovery
from torch import Tensor

def plot_stroke_points(x,y, start_points, square=False, freq=1):
    x_middle_strokes = x[np.where(start_points == 0)][::freq]
    y_middle_strokes = y[np.where(start_points == 0)][::freq]
    x_start_strokes = x[np.where(start_points == 1)]
    y_start_strokes = y[np.where(start_points == 1)]

    plt.scatter(x_middle_strokes, y_middle_strokes, s=1)

    max_y = np.max(y)
    head_length = .01*max_y
    head_width = .02*max_y
    x=x[::freq]; y=y[::freq]
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(zip(x, y),zip(x[1:], y[1:]))):
        if start_points[1:][i]:
            continue
        xdiff = (x2 - x1)
        ydiff = (y2 - y1)
        dx = min(xdiff / 2, max_y*.1) # arrow scaled according to distance, with max desired_num_of_strokes
        dy = min(ydiff / 2, max_y*.1)
        plt.arrow(x1, y1, dx, dy, color="blue", length_includes_head = True, head_length = head_length, head_width=head_width) # head_width = 1.4,

    plt.scatter(x_start_strokes, y_start_strokes, s=3)

pad_dpi = {"padding":.05, "dpi":71}

def render_points_on_image(gts, img, save_path=None, img_shape=None, origin='lower', invert_y_image=False, show=False, freq=1):
    return render_points_on_image_pil(gts, img, save_path, img_shape, origin, invert_y_image, show=show, freq=freq)

def render_points_on_image_matplotlib(gts, img_path, save_path=None, img_shape=None, origin='lower',
                                      invert_y_image=False, show=False, freq=1):
    """ This is for when loading the images created by matplotlib
    Args:
        gts: SHOULD BE (VOCAB SIZE X WIDTH)
        img_path:
        save_path:
        img_shape:

    Returns:

    """

    gts = np.array(gts)[:, ::freq]
    x = gts[0]
    y = gts[1]
    start_points = gts[2]

    if isinstance(img_path, str) or isinstance(img_path, Path):
        img_path = Path(img_path)
        img = cv2.imread(img_path.as_posix(), cv2.IMREAD_GRAYSCALE)
        img = img[::-1, :]

        if img_shape:
            scale_factor = img.shape[0]/60
            img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    else:
        img = img_path

    plt.imshow(img, cmap="gray", origin=origin)

    if True:
        ## PREDS: should already be scaled appropriately
        ## GTs: y's are scaled from 0-1, x's are "square"
        height = img.shape[0]
        # original images are 61 px tall and have ~7 px of padding, 6.5 seems to work better
        # this is because they are 1x1 inches, and have .05 padding, so ~.05*2*61
        x *= height * (1-6.5/61)
        y *= height * (1-6.5/61)

        ## 7 pixels are added to 61 pixel tall images;
        x += 6.5/61 * (pad_dpi["padding"]/.05) / 2 * height # pad_dpi["dpi"]
        y += 6.5/61 * (pad_dpi["padding"]/.05) / 2 * height

        plot_stroke_points(x,y,start_points, origin)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    if show:
        plt.show()

def render_points_on_image_pil(gts, img, save_path=None, img_shape=None, origin='lower',
                               invert_y_image=False, show=False, freq=1):
    """ This is for when drawing on the images created by PIL, which doesn't have padding
        Origin needs to be lower for the GT points to plot right

    Args:
        gts: SHOULD BE (VOCAB SIZE X WIDTH)
        img: Numpy representation, y-axis should already be reversed (origin='lower')
        save_path:
        img_shape:

    Returns:
    """
    img = np.asarray(img)
    if invert_y_image:
        img = img[::-1]
    gts = np.array(gts)
    height = img.shape[0]
    x = gts[0] * height
    y = gts[1] * height

    img_width_inches = int(img.shape[1] / height)
    plt.figure(figsize=(img_width_inches,2), dpi=200)
    plt.imshow(img, cmap="gray", origin=origin, interpolation="bicubic")

    start_points = gts[2] if gts.shape[0] > 2 else np.zeros(gts.shape[-1])
    plot_stroke_points(x,y,start_points, freq=freq)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    if show:
        plt.show()

def render_points_on_strokes(gts, strokes, save_path=None, x_to_y=None):
    gts = np.array(gts)
    x = gts[0]
    y = gts[1]
    start_points = gts[2]

    draw_strokes(normalize_stroke_list(strokes), x_to_y=x_to_y)
    plot_stroke_points(x,y,start_points)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def draw_strokes(stroke_list, x_to_y=1, line_width=None, save_path=""):
    def prep_figure(figsize=(5, 1), dpi=71):
        f = plt.figure(figsize=figsize, dpi=dpi)
        plt.axis('off')
        plt.axis('square')
        return f

    # plt.NullFormatter()
    if line_width is None:
        line_width = max(random.gauss(1, .5), .4)
    if x_to_y != 1 and not x_to_y is None:
        for stroke in stroke_list:
            stroke["x"] = [item * x_to_y for item in stroke["x"]]

    # Add tiny amount of padding
    y_min = min([min(x["y"]) for x in stroke_list])-.1
    y_max = max([max(x["y"]) for x in stroke_list])+.1
    x_min = min([min(x["x"]) for x in stroke_list])-.1
    x_max = max([max(x["x"]) for x in stroke_list])+.1

    if x_to_y:
        size = (ceil(x_to_y),1)
    else:
        size = (ceil((x_max-x_min)/(y_max-y_min)), 1)

    if save_path:
        f = prep_figure(figsize=size, dpi=pad_dpi["dpi"])

    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])

    for stroke in stroke_list:
        plt.plot(stroke["x"], stroke["y"], linewidth=line_width, color="black")

    if save_path:
        plt.savefig(save_path, pad_inches=pad_dpi["padding"], bbox_inches='tight') # adds 7 pixels total in padding for 61 height

    f.clear()
    plt.close(f)
    plt.close('all')


def draw_strokes_from_gt_list_OLD(stroke_list, x_to_y=1, line_width=None, save_path=""):
    def prep_figure(figsize=(5, 1), dpi=71):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.axis('off')
        plt.axis('square')

    # plt.NullFormatter()
    if line_width is None:
        line_width = max(random.gauss(1, .5), .4)
    if x_to_y != 1:
        for stroke in stroke_list:
            stroke["x"] = [item * x_to_y for item in stroke["x"]]

    if save_path:
        prep_figure(figsize=(ceil(x_to_y), 1), dpi=pad_dpi["dpi"])

    for stroke in stroke_list:
        plt.plot(stroke["x"], stroke["y"], linewidth=line_width, color="black")

    y_min = min([min(x["y"]) for x in stroke_list])
    y_max = max([max(x["y"]) for x in stroke_list])
    x_min = min([min(x["x"]) for x in stroke_list])
    x_max = max([max(x["x"]) for x in stroke_list])

    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])

    if save_path:
        plt.savefig(save_path, pad_inches=pad_dpi["padding"], bbox_inches='tight') # adds 7 pixels total in padding for 61 height
        plt.close()


def normalize_stroke_list(stroke_list, maintain_ratio=False):
    """ Max/min rescale to -1,1 range

    Args:
        my_array:

    Returns:

    """
    normalize = lambda _array,_max,_min: (((np.array(_array)-_min)/(_max-_min)-.5)*2).tolist()
    x_max = np.max([max(x["x"]) for x in stroke_list])
    x_min = np.min([min(x["x"]) for x in stroke_list])
    y_min = np.min([min(x["y"]) for x in stroke_list])
    y_max = np.max([max(x["y"]) for x in stroke_list])

    ## THIS DOES NOT MAINTAIN CENTERING!
    if maintain_ratio:
         xrange = x_max-x_min
         yrange = y_max-y_min
         x_max = xrange * yrange/xrange + x_min


    new_stroke_list = []
    for item in stroke_list:
        new_stroke_list.append({"x":normalize(item["x"].copy(), x_max, x_min), "y":normalize(item["y"].copy(), y_max, y_min)})

    return new_stroke_list

def normalize_gt(gt, left_pad=0, bottom_pad=0, top_pad=0):
    """ Rescales GT to 0,1 and 0,? maintaining aspect ratio
        MODIFIES VALUE
        There's no way to encode right padding in the GT
    """
    x_min, x_max, y_min, y_max = get_x_y_min_max_from_gt(gt)

    # Shift to zero, add in padding
    gt[:, 0] += -x_min + left_pad
    gt[:, 1] += -y_min + bottom_pad

    # Change y-axis to be on bigger scale
    y_max += top_pad + bottom_pad

    gt[:,0:2] = gt[:,0:2] / (y_max-y_min)
    return gt

def gt_to_raw(instance):
    start_points = np.array(stroke_recovery.relativefy(instance[:, 2]))
    start_indices = np.argwhere(start_points == 1).astype(int).reshape(-1)
    l = np.split(instance[:, 0:2], start_indices)
    output = []

    for item in l:
        if item.shape[0]:
            output.append({"x": item[:, 0].tolist(), "y": item[:, 1].tolist()})
    return output


def gt_to_list_of_strokes(instance, stroke_number=True, has_start_points=True, sos_error_check=False):
    """

    Args:
        instance: NUMPY!

    Returns:
        list of strokes LENGTH X (x,y) e.g. shape = [ (Len,2) , (stroke 2) ...
    """

    has_start_points = False if instance.shape[-1] <= 2 else True

    if has_start_points:
        # if start points are sequential 000011112222...

        if instance[0,2] != 1: # first point should be a start point
            warnings.warn("First SOS should be 1!!!")

        start_indices = stroke_recovery.get_sos_args(instance[:,2], stroke_numbers=stroke_number)
        start_points = stroke_recovery.relativefy(instance[:, 2]) if stroke_number else instance[:, 2]
        #np.argwhere(np.round(start_points) == 1).astype(int).reshape(-1)
        l = np.split(instance[:, 0:2], start_indices)
        if not l[0].size:
            l = l[1:]
        else:
            warnings.warn("First item should have been empty")
        if np.any(start_points < 0) and sos_error_check:
            raise Exception("Start points are less than 0")
        return l
    else:
        one_liner = [instance.flatten()]
        return one_liner

def get_x_y_min_max_from_gt(instance):
    x_max = np.max(instance[:, 0])
    x_min = np.min(instance[:, 0])
    y_min = np.min(instance[:, 1])
    y_max = np.max(instance[:, 1])
    #print(x_min, x_max, y_min, y_max)
    return x_min, x_max, y_min, y_max

def get_x_to_y_from_gt(instance, right_pad=0, top_pad=0):
    """
    Args:
        instance:
        right_pad (int): Include some padding at right of image
        top_pad (int): Include some padding at top of image

    Returns:

    """
    x_min, x_max, y_min, y_max = get_x_y_min_max_from_gt(instance)
    return (right_pad + x_max - x_min) / (top_pad + y_max - y_min)


def get_x_to_y_from_raw(instance):
    y_min = min([min(x["y"]) for x in instance])
    y_max = max([max(x["y"]) for x in instance])
    x_min = min([min(x["x"]) for x in instance])
    x_max = max([max(x["x"]) for x in instance])
    #print(y_min, y_max, x_min, x_max)
    return (x_max - x_min) / (y_max - y_min)


def draw_from_raw(raw, show=True, save_path=None, height=61, right_padding="random"):
    """ Raw is a list of strokes: {"x":(x1,x2,...), "y":(y1,y2,...)}
        Assumes regular normalization, 0,X and 0,1

    Args:
        raw:
        save_path:
        height:
    Returns:

    """
    if isinstance(right_padding, str):
        right_padding = np.random.randint(10)

    #x_to_y = get_x_to_y_from_raw(raw)
    #width = ceil(x_to_y * height)
    x_max = max([max(x["x"]) for x in raw])
    width = ceil(x_max) * height + right_padding

    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    color = 0
    linewidth = 2

    for line in raw:
        coords = zip((np.array(line["x"]) * height), (np.array(line["y"]) * height))
        line = list(coords)

        if line.size > 2:
            line = [tuple(x) for x in line.flatten().reshape(-1, 2).tolist()]
            draw.line(line, fill=color, width=linewidth, joint='curve')
        elif line.size == 2:
            line1 = line - linewidth / 2
            line2 = line + linewidth / 2
            line = np.r_[line1, line2].flatten().tolist()
            draw.ellipse(line, fill='black', outline='black')

    data = np.array(img)[::-1]  # invert the y-axis

    img = Image.fromarray(data, 'L')
    if save_path:
        img.save(save_path)
    if show:
        img.show()
    return data

def gt_to_pil(gt, stroke_number=False):
    """

    Args:
        gt:
        stroke_number:

    Returns:
        Pil format; list of strokes; each stroke is list of (x,y) points
                    e.g. [[(0,0),(1,1)], ]
    """

    list_of_strokes = gt_to_list_of_strokes(gt, stroke_number=stroke_number)
    for stroke in list_of_strokes:
        yield [tuple(stroke_point) for stroke_point in stroke.flatten().reshape(-1, 2).tolist()]

def rnd_width(w, on=True):
    if on and random.randint(0,1): # don't usually change the width
        w = w + random.randint(-1,1)
    return max(1, w)

def draw_from_gt(gt, show=True, save_path=None, min_width=None, height=61,
                 right_padding="random", linewidth=None, max_width=5, color=0, alpha=False,
                 use_stroke_number=None, plot_points=False, bonus_points=None,
                 x_rel=False,
                 **kwargs):
    """ RETURNS DATA IN "LOWER" origin format!!!
        GT is a WIDTH x VOCAB size numpy array
        Start strokes are inferred by [:,2], which should be 1 when the point starts a new stroke
        [:,0:2] are the x,y coordinates

    Args:
        raw:
        save_path:
        height:
        use_stroke_number: True - strokes labelled like 1,1,1,1,2,2,2,2; False - 00000100001
        bonus_points: other points to plot (e.g. intersections); should be [(x1,y1),(x2,y2)...]
    Returns:

    """
    # Make it 3 channels if using bonus points
    # if bonus_points and color==0:
    #     color = 0,0,0

    ### HACK
    if use_stroke_number is None:
        use_stroke_number = True if gt.shape[-1] > 2 and np.any(gt[:,2] >= 2) else False

    if isinstance(gt, Tensor):
        gt = gt.numpy()

    if isinstance(color, int):
        color = color,
    else:
        color = tuple(color)
    channels = len(color)
    image_type = "L" if channels == 1 else "RGB"
    background = tuple([255]*channels)
    if alpha:
        image_type += "A"
        color = tuple((*color, 255))
        background = tuple((*background, 0))

    using_random_width = True if linewidth is None else False
    if linewidth is None:
        linewidth = min(max(int(abs(np.random.beta(2,4)) * max_width + .8), 1),max_width)
        #min(max(int(abs(np.random.randn()) * (max_width - 1) * .5 + 1), 1),max_width)

    if isinstance(right_padding, str):
        right_padding = np.random.randint(6)

    # Put in absolute space
    if x_rel:
        gt = gt.copy()
        gt[:, 0] = np.cumsum(gt[:, 0])

    if np.isnan(gt).any():
        assert not np.isnan(gt).any()
    width = ceil(np.max(gt[:, 0]) * height) + right_padding
    width = max(width, height) # needs to be positive
    rescale = height

    if min_width:
        width = max(width, min_width)

    # else: # If a width is specified, we can't rescale to height
    #     max_rescale = min_width / np.max(gt[:, 0])
    #     rescale = min(height, max_rescale)

    gt_rescaled = np.c_[gt[:, 0:2] * rescale, gt[:, 2:]]
    pil_format = gt_to_list_of_strokes(gt_rescaled, stroke_number=use_stroke_number)
    img = Image.new(image_type, (width, height), background)
    draw = ImageDraw.Draw(img)

    #sos_args = stroke_recovery.get_sos_args(gt_rescaled, stroke_numbers=use_stroke_number)
    _color = color
    for i, line in enumerate(pil_format):
        if line.size > 2:
            line = [tuple(x) for x in line.flatten().reshape(-1, 2).tolist()]
            if len(line) > 20 and using_random_width: # make some lines change widths mid-line
                split = random.randint(10, len(line)-10)
                draw.line(line[:split], fill=_color, width=rnd_width(linewidth, using_random_width), joint='curve')
                draw.line(line[split:], fill=_color, width=rnd_width(linewidth, using_random_width), joint='curve')
            else:
                draw.line(line, fill=_color, width=rnd_width(linewidth, using_random_width), joint='curve')
        elif line.size == 2: # only have a single coordinate, make it big!
            line1 = line - linewidth / 2
            line2 = line + linewidth / 2
            line = np.r_[line1, line2].flatten().tolist()
            draw.ellipse(line, fill=_color, outline=color)

    if plot_points:
        image_type = "RGB"
        stroke_point_size=2
        background = Image.new(image_type, (width, height), (255, 255, 255))
        background.paste(img)  # 3 is the alpha channel
        draw = ImageDraw.Draw(background)

        for line in pil_format:
            for i, point in enumerate(line):
                color = 'blue' if i else 'orange'
                line1 = point - stroke_point_size / 2
                line2 = point + stroke_point_size / 2
                point = np.r_[line1, line2].flatten().tolist()
                draw.ellipse(point, fill=color, outline=color)
        img = background

    if not bonus_points is None:
        for point in np.asarray(bonus_points):
            line1 = point - linewidth / 2
            line2 = point + linewidth / 2
            #line = tuple(np.r_[line1, line2].flatten().tolist())
            line = (tuple(line1),tuple(line2))
            draw.ellipse(line, fill=(255,0,255), outline=(255,0,255))

    data = np.array(img)[::-1]  # invert the y-axis, to upper origin

    img = Image.fromarray(data, image_type)

    if save_path:
        img.save(save_path)
    if show:
        img.show()

    return data

def random_paired_pad(gt, img, vpad=10, hpad=10, height=61):
    """ DEPRECATED Not really needed for anything, mostly verifies we're padding GT's/images the same

    Note: GTs assume normal origin
              Imgs use top left origin, should reverse first
    Args:
        gt: A GT (WIDTH X VOCAB) with x,y in :,0:2
        img: 1 channel numpy pixel array
        vpad (int): Random amount to pad on each side
        hpad (int): Random amount to pad on each side
        height: Assumed height in pixels!! With the image, we add this many pixels, with the GT,
                    everything should be multiplied by height to get into pixel space
    Returns:
        gt, img
    """
    from skimage.transform import resize

    lpad = np.random.randint(hpad)
    rpad = np.random.randint(hpad)
    tpad = np.random.randint(vpad)
    bpad = np.random.randint(vpad)
    print("Top", tpad)
    print("Bottom", bpad)
    print("Right", rpad)
    print("Left", lpad)

    # y-dimension is reversed for images
    new_img = np.pad(img, ((tpad, bpad), (lpad, rpad)), constant_values=255)

    # Downsize
    new_img = resize(new_img, (height, 61*10))


    # Rescale new_gt
    new_gt = normalize_gt(gt.copy(), left_pad=lpad/height, right_pad=rpad/height, bottom_pad=bpad/height, top_pad=tpad/height)
    x_min, x_max, y_min, y_max = get_x_y_min_max_from_gt(new_gt)
    print(x_min, x_max, y_min, y_max)

    return new_gt, new_img


def random_pad(gt, vpad=10, hpad=10, height=61):
    """
    Note: GTs assume normal origin
              Imgs use top left origin, should reverse first

          There is no such thing as a right pad for these pictures!

    Args:
        gt: A GT (WIDTH X VOCAB) with x,y in :,0:2
        vpad (int): Random amount to pad on each side
        hpad (int): Random amount to pad on each side
        height: Assumed height in pixels!! With the image, we add this many pixels, with the GT,
                    everything should be multiplied by height to get into pixel space
    Returns:
        gt, img
    """
    lpad = np.random.randint(hpad)
    tpad = np.random.randint(vpad)
    bpad = np.random.randint(vpad)

    # print("Top", tpad)
    # print("Bottom", bpad)
    # print("Left", lpad)

    # Rescale new_gt
    new_gt = normalize_gt(gt, left_pad=lpad/height, bottom_pad=bpad/height, top_pad=tpad/height)

    return new_gt


def overlay_images(background_img=None, foreground_gt=None, normalized=True, save_path=None,
                   color=[255,0,0],
                   linewidth=1):
    """

    Args:
        background_img: Should be normalized 0-1, numpy!
        foreground_gt:

    Returns:

    """
    rescale = lambda x: (x+1)*127.5 if normalized else lambda x: x
    ## PLOT THE RED LINE VERSION
    if background_img is not None:
        img = Image.fromarray(np.uint8(rescale(np.squeeze(background_img))), 'L')  # actual image given to model
        img = img.convert("RGB")
        if foreground_gt is None:
            return img
    # elif isinstance(background_img, np.ndarray):
    #     img = Image.fromarray(background_img, "RGB")

    if foreground_gt is not None:
        red_img = draw_from_gt(foreground_gt, show=False, linewidth=linewidth, color=color, alpha=True)
        red_img = Image.fromarray(np.uint8(red_img), 'RGBA')
        if background_img is None:
            return red_img

    height = max(img.size[1], red_img.size[1])
    width = max(img.size[0], red_img.size[0])

    # Make new white image
    bg = Image.new('RGB', (width, height), (255, 255, 255))
    bg.paste(img, (0, 0))
    bg.paste(red_img, (0, 0), red_img)
    if save_path:
        bg.save(save_path)
    return bg


if __name__ == "__main__":
    test_conv_weight()
    # test_gt_stroke_length_generator()
    Stop
    os.chdir("../data")
    with open("online_coordinate_data/3_stroke_16_v2/train_online_coords.json") as f:
        output_dict = json.load(f)

    instance = output_dict[11]
    render_points_on_image(instance['gt'], img=instance['image_path'], x_to_y=instance["x_to_y"])
