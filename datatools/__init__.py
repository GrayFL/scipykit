from . import dataset

# import os
# import cv2
# import numpy as np

# class Sample():

#     COLOR_GREEN = (0,255,0)
#     COLOR_BLUE = (255,0,0)
#     COLOR_WHITE = (255,255,255)
#     COLOR_BLACK = (0,0,0)
#     LABEL_DICT = {
#         0:'ignored',
#         1:'pedestrian',
#         2:'people',
#         3:'bicycle',
#         4:'car',
#         5:'van',
#         6:'truck',
#         7:'tricycle',
#         8:'awning-tricycle',
#         9:'bus',
#         10:'motor',
#         11:'others'
#     }

#     def __init__(
#             self, img:np.ndarray|str, ann:list[str]|str,
#             selection:list[int] = None):
#         if type(img)==str:
#             self.img = cv2.imread(img)
#         else:
#             self.img = img
#         if type(ann)==str:
#             with open(ann) as f:
#                 self.ann = np.fromstring(
#                     f.read().replace('\n',','),dtype='uint',sep=',').reshape(-1,8)
#         else:
#             self.ann = np.fromstring(ann,dtype='uint',sep=',').reshape(-1,8)
#         if selection != None:
#             self.ann = self.ann[np.isin(self.ann[:,5],selection)]
#             # 只摘取指定的类别
#         if len(self.ann) > 0:
#             self.scale = self.ann[:,2:4].mean(axis=0)
#         else:
#             self.scale = np.array([0,0])
#         self.ann[:,2]+=self.ann[:,0]
#         self.ann[:,3]+=self.ann[:,1]

#     def disp(
#             self, selection:list[int] = None, label:bool = True,
#             box_fill:bool = True, box_thickness:int = 2,
#             fontScale:float = 0.4, font_thickness:int = 1,
#             RGB:bool = True,
#             alpha:float = 0.5, replace:bool = False,
#             )->np.ndarray:
#         f"""
#         selection:
#         {Sample.LABEL_DICT}
#         RGB: 使用RGB输出，但不改变Sample本身
#         """
#         output = self.img.copy()
#         blk = np.zeros(output.shape, dtype='uint8')
#         for bbox in self.ann:
#             if selection != None and bbox[5] not in selection:
#                 continue
#             if box_fill:
#                 cv2.rectangle(
#                     blk, bbox[0:2],
#                     bbox[2:4],
#                     Sample.COLOR_GREEN,
#                     thickness=-1,
#                     )
#             cv2.rectangle(
#                 blk, bbox[0:2],
#                 bbox[2:4],
#                 Sample.COLOR_BLUE,
#                 box_thickness,
#                 )
#             if label:
#                 (base_w,base_h),bottom = cv2.getTextSize(
#                     f'{Sample.LABEL_DICT[bbox[5]]}',
#                     fontFace=cv2.FONT_HERSHEY_DUPLEX,
#                     fontScale=fontScale,
#                     thickness=font_thickness,
#                 )
#                 # print(bbox,base_w,base_h)
#                 cv2.rectangle(
#                     blk,
#                     (bbox[0],bbox[1]-base_h),
#                     (bbox[0]+base_w,bbox[1]+bottom),
#                     Sample.COLOR_BLUE,
#                     thickness=-1,
#                 )
#                 cv2.putText(
#                     blk,
#                     f'{Sample.LABEL_DICT[bbox[5]]}',
#                     bbox[0:2],
#                     fontFace=cv2.FONT_HERSHEY_DUPLEX,
#                     color=Sample.COLOR_WHITE,
#                     fontScale=fontScale,
#                     thickness=font_thickness,
#                     # bottomLeftOrigin=True
#                 )
#         if label:
#             output = cv2.addWeighted(output,1,blk,alpha,1)
#         # output = alpha*blk + output
#         if replace:
#             self.img = output
#         if RGB:
#             output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
#         return output

#     def resize_ann(
#             self, factor:float = 1, replace = False
#         )->np.ndarray:
#         tmp = self.ann.copy()
#         tmp[:,:4] = (tmp[:,:4] * factor).astype('uint')
#         if replace:
#             self.ann = tmp
#         return tmp

#     def __repr__(self) -> str:
#         return f'包含标注数量：{len(self.ann)}'

# class Dataset():

#     def __init__(self, arr_sample:list[Sample] = []):
#         self.arr_sample = arr_sample

#     def __getitem__(self,index)->Sample:
#         return self.arr_sample[index]

#     def __len__(self)->int:
#         return len(self.arr_sample)

#     def build_dataset(
#             self, IMG_FILE_PATH:str, ANN_FILE_PATH:str,
#             max_sample_len:int=None, **kwds
#             )->list[Sample]:
#         self.arr_sample.clear()
#         arr_img_file_name = os.listdir(IMG_FILE_PATH)
#         arr_ann_file_name = os.listdir(ANN_FILE_PATH)
#         if max_sample_len != None:
#             sample_len = min(len(arr_img_file_name),max_sample_len)
#         else:
#             sample_len = len(arr_img_file_name)
#         for i in range(sample_len):
#         # for img_file_name, ann_file_name in arr_img_file_name, arr_ann_file_name:
#             self.arr_sample.append(
#                 Sample(
#                     IMG_FILE_PATH+arr_img_file_name[i],
#                     ANN_FILE_PATH+arr_ann_file_name[i],
#                     **kwds
#                 )
#             )
#         return self.arr_sample
