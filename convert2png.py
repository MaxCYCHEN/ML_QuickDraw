import json
import os
import datetime as dt
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import ast
from PIL import Image


def f2cat(filename: str) -> str:
    return filename.split('.')[0]

class convert2png():
    def __init__(self, input_path='dataset/train_simplified_5/',
                  output_path='dataset/simplified_small_bitmap_64/'):
        self.input_path = input_path
        self.output_path = output_path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            os.makedirs(os.path.join(self.output_path, 'train'))
            os.makedirs(os.path.join(self.output_path, 'validation'))
            os.makedirs(os.path.join(self.output_path, 'test'))
        else:
            print('Work folder exist!')

    def list_all_categories(self):
        #files = os.listdir(os.path.join(self.input_path, 'train_simplified'))
        files = os.listdir(os.path.join(self.input_path))
        return sorted([f2cat(f) for f in files], key=str.lower)
    
    def read_training_csv(self, category, nrows=None, usecols=None, drawing_transform=False):
        df = pd.read_csv(os.path.join(self.input_path, category + '.csv'),
                         nrows=nrows, parse_dates=['timestamp'], usecols=usecols)
        if drawing_transform:
            df['drawing'] = df['drawing'].apply(json.loads)
        return df

    def create_label_folder(self, folder_path, cat):
        if not os.path.exists(os.path.join(folder_path, cat)):
            os.makedirs(os.path.join(folder_path, cat))

    def draw_cv2_noresize(self, raw_strokes, size=256, lw=1, time_color=True):

        #line width
        # lw = np.round((size / BASE_IMG_SIZE) * 6.0).astype(np.int8)
    
        # Define a function to add 1 to each element
        x = 0
        def add_one_to_element(element):
            return np.round((int(element) / 255.) * (size - 1)).astype(np.int8)
        
        # Use map to apply the function to each element in the nested list
        result_strokes = list(map(lambda inner_list: list(map(lambda sublist: list(map(add_one_to_element, sublist)), inner_list)), raw_strokes))
    
        img = np.zeros((size, size), np.uint8)
        for t, stroke in enumerate(result_strokes):
            for i in range(len(stroke[0]) - 1):
                color = 255 - min(t, 10) * 13 if time_color else 255 # encode the color by line sequence 
                _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    
        return img        

    def image_generator(self, plt_path, df, size, lw, time_color):
        x = np.zeros((len(df), size, size, 1))
        for i, raw_strokes in tqdm(enumerate(df.values)):
                #print(i, raw_strokes)
                img = self.draw_cv2_noresize(raw_strokes, size=size, lw=lw, time_color=time_color)
                im = Image.fromarray(img)
                im.save(os.path.join(plt_path, '{}.png'.format(i)))
                #x[i, :, :, 0] = img

        # debug
        if 0:
            n = 8
            fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
            for i in range(n**2):
                ax = axs[i // n, i % n]
                ax.imshow(x[i, :, :, 0], cmap=plt.cm.gray)
                ax.axis('off')
            plt.tight_layout()
            fig.savefig('test.png', dpi=300)
            plt.show(); 
            

                       

# Main
s = convert2png()
categories = s.list_all_categories()
print("Total categories: {}".format(len(categories)))

IMG_SIZE = 64 # The size of output img
NROWS = 100000 # Total data per categories
VAL_PEC = 0.1
TEST_PEC = 0.1

for y, cat in tqdm(enumerate(categories)):
    df = s.read_training_csv(cat, nrows=NROWS)
    #print(df[0:10])
    
    s.create_label_folder(os.path.join(s.output_path, 'train'), cat)
    s.create_label_folder(os.path.join(s.output_path, 'validation'), cat)
    s.create_label_folder(os.path.join(s.output_path, 'test'), cat)
    
    # Shuffle the data
    df_shuffled = df['drawing'].sample(frac=1, random_state=29).apply(ast.literal_eval)
    #df_shuffled = df['drawing'].apply(ast.literal_eval)

    
    # Partial the data into train, test, val
    part_val_size = int(np.round(VAL_PEC * NROWS))
    part_test_size = int(np.round(TEST_PEC * NROWS))
    part_train_size = NROWS - part_val_size - part_test_size
    part_train = df_shuffled.iloc[:part_train_size]
    part_val = df_shuffled.iloc[part_train_size:part_train_size + part_val_size]
    part_test = df_shuffled.iloc[part_train_size + part_val_size:]
    #print(len(part_train),len(part_val),len(part_test))
    #print(part_train.values[0][0][0])

    # Convert drawing strokes to bitmap
    s.image_generator(os.path.join(s.output_path, 'train', cat), part_train, IMG_SIZE, 1, False)
    s.image_generator(os.path.join(s.output_path, 'validation', cat), part_val, IMG_SIZE, 1, False)
    s.image_generator(os.path.join(s.output_path, 'test', cat), part_test, IMG_SIZE, 1, False)   
