import sys
import numpy as np
import cv2
from PIL import Image as PILImage, ImageDraw, ImageFont
from rembg import remove

from detect_info import DetectInfo

# font = ImageFont.truetype('Arial.ttf', 20)
# arrow_font = ImageFont.truetype('Arial.ttf', 50)

# font = ImageFont.truetype('Humor-Sans.ttf', 20)
# arrow_font = ImageFont.truetype('Humor-Sans.ttf', 50)

font = None
arrow_font = None

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
    color=(0, 0, 255)
    # color=(255, 0, 0) #red
    font = cv2.FONT_HERSHEY_SIMPLEX
    x1, y1, x2, y2 = coord
    lu_x, lu_y, rb_x, rb_y = coord
    
    cv2.rectangle(img, pt1=(int(lu_x), int(lu_y)), pt2=(int(rb_x), int(rb_y)), color=color, thickness=1)
    cv2.putText(img, label, org=(int(lu_x) - 15, int(lu_y) - 20), fontFace=font, fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)

    if has_left:
        x = int(lu_x - 30)
        # print(rb_y, lu_y)
        y = int(lu_y + ((rb_y - lu_y)/2) + 10)
        cv2.putText(img, text="<", org=(x, y), fontFace=font, color=color, thickness=2, fontScale=1, lineType=cv2.LINE_AA)

    if has_right:
        x = int(rb_x + 10)
        # print(rb_y, lu_y)
        y = int(rb_y - ((rb_y - lu_y)/2) + 10)
        cv2.putText(img, text=">", org=(x, y), fontFace=font, color=color, thickness=2, fontScale=1, lineType=cv2.LINE_AA)

    return img
    

def nparr_to_img(arr: np.ndarray) -> PILImage:
    return PILImage.fromarray(arr)

def img_to_nparr(img: PILImage) -> np.ndarray:
    return np.array(img)


def cal_size(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # _tmp = Image.fromarray(img)
    # _tmp.show()

    height, width = img.shape

    ret, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)# 閾値で2値化
    # new_img = cv2pil(binary)
    # new_img.show()
    
    # _tmp = Image.fromarray(binary)
    # _tmp.show()

    obj_size = np.count_nonzero(binary == 255)
    return obj_size

def cal_size_using_bg(img: np.ndarray):
    resize_rate = 0.5
    _img = np.copy(img)
    _img = cv2.resize(frame, dsize=None, fx=resize_rate, fy=resize_rate)
    _img = remove(_img)
    _img = cv2.resize(frame, dsize=None, fx=1/resize_rate, fy=1/resize_rate)
    _size = np.count_nonzero(_img != 0)
    
    # _tmp = PILImage.fromarray(_img)
    # _tmp.show()
    return _size


def get_left(src: DetectInfo, targets: list[DetectInfo]):
    src_x, _,_,_ = src.box

    targets_in_left_side = []
    for t in targets:
        target_x,_,_,_ = t.box
        if target_x < src_x:
            targets_in_left_side.append(t)


    left_target = None
    d = sys.maxsize
    for t in targets_in_left_side:
        x,_,_,_ = t.box
        if x < d:
            d = x
            left_target = t
    return left_target

def get_right(src: DetectInfo, targets: list[DetectInfo]):
    _,  _, src_x, _ = src.box

    targets_in_right_side = []
    for t in targets:
        target_x, _, _ ,_ = t.box
        if target_x > src_x:
            targets_in_right_side.append(t)

    right_target = None
    d = sys.maxsize
    for t in targets_in_right_side:
        _, _, x, _ = t.box
        if x < d:
            d = x
            right_target = t
            
    return right_target



def set_arrow(items: list[DetectInfo]):
    for item in items:
        left = get_left(item, items)            
        if left is not None and left.size > item.size:
            item.left_arrow = True

        right = get_right(item, items)            
        if right is not None and right.size < item.size:
            item.right_arrow = True
    return items


    
    

def crop(img: np.ndarray, lu_x: int, lu_y:int, rb_x:int, rb_y:int):
    _img = np.copy(img)
    return _img[lu_y:rb_y, lu_x:rb_x, :]

def iou(a: tuple, b: tuple) -> float:
    # https://ohke.hateblo.jp/entry/2020/06/20/230000
    a_x1, a_y1, a_x2, a_y2 = a
    b_x1, b_y1, b_x2, b_y2 = b
    
    if a == b:
        return 1.0
    elif (
        (a_x1 <= b_x1 and a_x2 > b_x1) or (a_x1 >= b_x1 and b_x2 > a_x1)
    ) and (
        (a_y1 <= b_y1 and a_y2 > b_y1) or (a_y1 >= b_y1 and b_y2 > a_y1)
    ):
        intersection = (min(a_x2, b_x2) - max(a_x1, b_x1)) * (min(a_y2, b_y2) - max(a_y1, b_y1))
        union = (a_x2 - a_x1) * (a_y2 - a_y1) + (b_x2 - b_x1) * (b_y2 - b_y1) - intersection
        return intersection / union
    else:
        return 0.0


def nms(items: list[DetectInfo], iou_threshold: float) -> list:
    # https://ohke.hateblo.jp/entry/2020/06/20/230000
    new_items = []
    scores = [v.score for v in items]
    
    while len(items) > 0:
        i = scores.index(max(scores))
        new_item = items.pop(i)
        scores.pop(i)
        
        delete_items = []
        delete_scores = []        
        for j, item_j in enumerate(items):
            iou_v = iou(new_item.box, item_j.box)
            if iou_v > iou_threshold:
                delete_items.append(items[j])
                delete_scores.append(scores[j])

        for item, score in zip(delete_items, delete_scores):
            items.remove(item)
            scores.remove(score)
            
        new_items.append(new_item)
    return new_items
  


class ArrangementArrow():
    def __init__(self, display_width: int, display_height: int):
        self.__font = cv2.FONT_HERSHEY_SIMPLEX
        self.__color_red=(255, 0, 0)
        self.__color_blue=(0, 0, 255)

        unit = int(display_width/3)
        y = int(display_height/3)
        self.__arrow = {
            "small": {
                "pt1": (int(unit/2), y),
                "pt2": (int(unit/2), 30),
                "current": False
            },
            "middle": {
                "pt1": (int(3/2*unit), y),
                "pt2": (int(3/2*unit), 30),
                "current": False
            },
            "large": {
                "pt1": (int(5/2*unit), y),
                "pt2": (int(5/2*unit), 30),
                "current": False  
            }
        }
        # self.__text = "here"
        self.__SMALL = 100000
        self.__MIDDLE = 150000
        
    def clean_arrow(self):
        for k in ["small", "middle", "large"]:
            self.__arrow[k]["current"] = False

    def set_arrow(self, frame: np.ndarray, item: DetectInfo):
        if item.size < self.__SMALL:
            self.__arrow["small"]["current"] = True
            
        if self.__SMALL <= item.size < self.__MIDDLE:
            self.__arrow["middle"]["current"] = True
            
        if self.__MIDDLE <= item.size:
            self.__arrow["large"]["current"] = True
        
        for key in self.__arrow:
            color =  self.__color_red if self.__arrow[key]["current"] else self.__color_blue
            cv2.arrowedLine(frame,
                            pt1=self.__arrow[key]["pt1"],
                            pt2=self.__arrow[key]["pt2"],
                            color=color,
                            thickness=20,
                            shift=0,
                            tipLength=0.6
                          )

            # cv2.putText(frame,
            #             text=self.__text,
            #             org=self.__arrow[key]["org"],
            #             fontFace=self.__font,
            #             color=color,
            #             thickness=2,
            #             fontScale=10,
            #             lineType=cv2.LINE_AA)
        
  
    
  
    
    

