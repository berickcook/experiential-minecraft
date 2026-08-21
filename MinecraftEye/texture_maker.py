import numpy as np
from PIL import Image
import csv

raw = list(csv.reader(open('assets/minecraft_textures_atlas_blocks.png.txt', 'r'), delimiter='\t'))
clean = list()
base = Image.open('assets/minecraft_textures_atlas_blocks.png_0.png')

for item in raw:
    if 'minecraft:block/' in item[0]:
        item[0] = item[0][16::]
        clean.append(item)

width = 48
height = len(clean) * 16

new_img = Image.new('RGB', (width, height), (0,0,0,0))

for i, item in enumerate(clean):
    top_x = int(item[1][2::])
    top_y = int(item[2][2::])
    new_text = base.crop((top_x, top_y, top_x + 16, top_y + 16))

    new_img.paste(new_text,(0, i * 16, 16, (i * 16) + 16), new_text)
    new_img.paste(new_text, (16, i * 16, 32, (i * 16) + 16), new_text)
    new_img.paste(new_text, (32, i * 16, 48, (i * 16) + 16), new_text)

new_img.save('assets/texture_array.png')

# list_im = ['Test1.jpg', 'Test2.jpg', 'Test3.jpg']
# imgs    = [ Image.open(i) for i in list_im ]
# # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
# min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
# imgs_comb = np.hstack([i.resize(min_shape) for i in imgs])
#
# # save that beautiful picture
# imgs_comb = Image.fromarray( imgs_comb)
# imgs_comb.save( 'Trifecta.jpg' )
#
# # for a vertical stacking it is simple: use vstack
# imgs_comb = np.vstack([i.resize(min_shape) for i in imgs])
# imgs_comb = Image.fromarray( imgs_comb)
# imgs_comb.save( 'Trifecta_vertical.jpg' )