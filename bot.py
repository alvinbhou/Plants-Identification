# !/usr/bin/python 
# -*-coding:utf-8 -*- 
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# from scipy.misc import imread, imsave, imresize
from PIL import Image
from scipy import signal
import time, datetime, random
import re, requests, json
import numpy as np
import asyncio
import telepot
from telepot.aio.loop import MessageLoop
from telepot.aio.delegate import per_chat_id, create_open, pave_event_space, include_callback_query_chat_id
from telepot.namedtuple import InlineQueryResultArticle, InputTextMessageContent
from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove, KeyboardButton
from keras.models import Model, load_model
from vis.visualization import visualize_saliency, visualize_cam, visualize_activation

mDict = json.load(open('./plants_dict.json'))
model_path = "./model/ResNet50_t3_2.h5"
model = load_model(model_path)

class User:
    def __init__(self, chatid):        
        self.chat_id = chatid

def getUser(chat_id):
    for user in users:
        if user.chat_id == chat_id:
            return user
    return None

def formatMsg(msg):
    # info = '我們預測該照片可能是\n'
    info = 'Most likely result..\n'
    reply =  info + '『' + str(msg) + '』'
    return reply

def formatTop5Msg(result):
    reply = '[Top5]\n'
    for r in result:
       reply += r[0] + ' : ' + str(r[1]) + '\n'
    return reply

def boxing(img, label):
 
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == "dense_2"][0]
    heatmap = visualize_saliency(model, layer_idx, np.expand_dims(label, axis=0), img)
    k_size = 28
    k = np.ones((k_size,k_size))/k_size
    heatmap = signal.convolve2d(heatmap[:,:,0], k, boundary='wrap', mode='same')/k.sum()
    threshold = heatmap.max() * 0.3
    maxTop = maxLeft = 999999999
    maxRight = maxBottom = -1
    for h in range(224):
        for w in range(224):
            # print(h,w)
            if heatmap[h][w] > threshold:
                if h < maxTop: maxTop = h
                if h > maxBottom: maxBottom = h
                if w < maxLeft: maxLeft = w
                if w > maxRight: maxRight = w


    maxTop = int(maxTop/3)
    maxBottom = int(maxBottom/3)
    maxLeft = int(maxLeft/3)
    maxRight = int(maxRight/3)
    img = img.copy()
    for h in range(224):
        for w in range(224):
            if (int(h/3) == maxTop and int(w/3) in range(maxLeft, maxRight)) or (int(h/3) == maxBottom and int(w/3) in range(maxLeft, maxRight)) or (int(w/3) == maxRight and int(h/3) in range(maxTop, maxBottom))  or (int(w/3) == maxLeft and int(h/3) in range(maxTop, maxBottom)):
                img[h][w][0] = img[h][w][1] = 255
                img[h][w][2] = 0

    return img

users = [] 
service_keyboard = ReplyKeyboardMarkup(
                            keyboard=[
                                [KeyboardButton(text="Feeling lucky!"),KeyboardButton(text="Help")], 
                            ]
                        ) 
def getKeybyVal(mDict,idx):
    return list(mDict.keys())[list(mDict.values()).index(idx)]


def getSampleImages():
    imgs = [f for f in os.listdir('sample-img') if os.path.isfile(os.path.join('sample-img', f))]
    return imgs

class PRBot(telepot.aio.helper.ChatHandler):

    def __init__(self, *args, **kwargs):
        super(PRBot, self).__init__(*args, **kwargs)

    async def on_chat_message(self, msg):      
        content_type, chat_type, chat_id = telepot.glance(msg)

        if content_type == 'photo':
            # download image and predict
            await bot.download_file(msg['photo'][-1]['file_id'], 'img/tmpImg.png')
            img = Image.open('img/tmpImg.png')
            img = img.resize((224,224), Image.BILINEAR)
            img = np.asarray(img)
            prob = model.predict(np.expand_dims(img, axis=0))
            
            # get top 5
            sorted_idx = list(np.argsort(prob[0]))
            sorted_idx = sorted_idx[::-1]
            top_result = getKeybyVal(mDict, sorted_idx[0])
            top5_idx = sorted_idx[0:5]
            top5_result = []
            for idx in top5_idx:
                top5_result.append([getKeybyVal(mDict, idx),  '{:.4%}'.format(prob[0][idx])])

            # boxing
            bbox_img = boxing(img,sorted_idx[0])
            save_img_name = '.' + str(chat_id) + '_bbox_img.png'
            result = Image.fromarray(bbox_img)
            result.save(os.path.join('img', save_img_name))
            # send result
            await self.sender.sendPhoto(open(os.path.join('img', save_img_name), 'rb')) 
            await self.sender.sendMessage(formatMsg(top_result))
            await self.sender.sendMessage(formatTop5Msg(top5_result), reply_markup=service_keyboard)

            os.remove(os.path.join('img', save_img_name))
            os.remove(os.path.join('img', 'tmpImg.png'))
            return
        elif content_type == 'text':
            if(getUser(chat_id) is None):
                print("new user", chat_id)
                user = User(chat_id)
                users.append(user)

            msg = msg['text']
            print(chat_id, msg) 

            if msg == '/start':
                await self.sender.sendMessage( "您好！請隨意上傳照片會進行植物分類預測\n Hi there! Upload any photo and see the classification result! :)", reply_markup=service_keyboard)
            elif msg == 'Help' or msg == '/help':
                await self.sender.sendMessage( "Project Github: https://github.com/CryoliteZ/Plants-Identification", reply_markup=service_keyboard)
            elif msg == 'Feeling lucky!':
                filename = random.choice(sampleImgs)
                img = imread( os.path.join('sample-img', filename), mode ='RGB')
                img = imresize(img ,size=(224,224))
                label = int(filename[:-4])
                result = getKeybyVal(mDict, label)
                bbox_img = boxing(img,(label))
                imsave('sample-img/bbox_img.png', bbox_img)
                await self.sender.sendPhoto(open('sample-img/bbox_img.png', 'rb')) 
                await self.sender.sendMessage( formatMsg(result), reply_markup=service_keyboard)
                os.remove(os.path.join('sample-img', 'bbox_img.png'))
        return

            
TOKEN = sys.argv[1]  # get token from command-line
sampleImgs = getSampleImages()
bot = telepot.aio.DelegatorBot(TOKEN, [
    include_callback_query_chat_id(
        pave_event_space())(
        per_chat_id(), create_open, PRBot, timeout= 120),
])

loop = asyncio.get_event_loop()
loop.create_task(MessageLoop(bot).run_forever())
print('Listening ...')
loop.run_forever()