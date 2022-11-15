import numpy as np
import cv2
from PIL import Image as PILImage, ImageDraw, ImageFont

font = ImageFont.truetype('Arial.ttf', 20)
arrow_font = ImageFont.truetype('Arial.ttf', 50)

# for pil
def draw(img: PILImage, coord: tuple[int], label: str, has_left: bool, has_right: bool) -> PILImage:
    """
    args:
    image: image
    coord: (left upper x,left upper y,right bottom x,right bottom y)
    """

    # color = (0,255,17) # green
    color = (255,0,0) # red

    
    draw = ImageDraw.Draw(img)
    draw.rectangle(coord, fill=None, outline=color, width=2)
    lu_x, lu_y, _, _ = coord
    draw.text((lu_x - 15, lu_y - 30), label, color, font=font)


    # if has_left:
    #     lu_x, lu_y, _, rb_y = coord
    #     x = lu_x - 50
    #     print(rb_y, lu_y)
    #     y = lu_y + ((rb_y - lu_y)/2) - 20
    #     draw.text((x, y), "←", color, font=arrow_font)

    # if has_right:
    #     _, l, rb_x, rb_y = coord
    #     x = rb_x + 10
    #     print(rb_y, lu_y)

    #     y = rb_y - ((rb_y - lu_y)/2) - 20
    #     draw.text((x, y), "→", color, font=arrow_font)


    return img


def draw_to_cv2(img: np.ndarray, coord: tuple[int], label: str, has_left: bool, has_right: bool) -> np.ndarray:
    color=(0, 255, 0)

    x1, y1, x2, y2 = coord
    
    cv2.rectangle(img, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=color, thickness=2)
    # cv2.putText(img, label, (x1 - 15, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * multiplier, (0, 0, 0), 1)
    return img
    

def nparr_to_img(arr: np.ndarray) -> PILImage:
    return PILImage.fromarray(arr)

def img_to_nparr(img: PILImage) -> np.ndarray:
    return np.array(img)

    
