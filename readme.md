# Plants-Identification
## Project Info
Plants identification on [240 categories](plants.csv) Taiwan endemic plants with a dataset size of 41834 images. Trained on ResNet50 with pre-trained weights from imagenet.

Designed an easy-to-use chatbot on Telegram for users to upload photos of plants and get instant boxing and predicted category result.

### Demo Telegram Chatbot
[<img src="http://icons.iconarchive.com/icons/froyoshark/enkel/64/Telegram-icon.png">](https://t.me/Plant_Classification_Bot) [Plant Classification Bot](https://t.me/Plant_Classification_Bot)

## Usage

plant_classifier.py
```
usage: plant_classfier.py [-h] --uid UID [--train_path TRAIN_PATH]
                          [--valid_path VALID_PATH] [--train_resnet]
                          [--train_inception] [--learning_rate LEARNING_RATE]
                          [--batch_size BATCH_SIZE] [--epoch EPOCH]
                          [--img_size IMG_SIZE]
optional arguments:
  -h, --help            show this help message and exit
  --uid UID             training uid
  --train_path TRAIN_PATH
                        training data path
  --valid_path VALID_PATH
                        valid data path
  --train_resnet        whether train on ResNet50
  --train_inception     whether train on InceptionResNetV2
  --learning_rate LEARNING_RATE
                        learning rate for training
  --batch_size BATCH_SIZE
                        batch size for training
  --epoch EPOCH         epochs for training
  --img_size IMG_SIZE   img width, height size
```      
bot.py
```
usage: python bot.py TELEGRAM_KEY_TOKEN
```                  

  

## Results

### Classification results

![](/img/ResNet50_f_uf_log_acc.png)

Training accuracy on 33372 images: 0.744375

Testing accuracy on 8374 images: 0.64296875

Testing Top 5 accuracy 8374 images: 0.81634120


### Boxing results
Example on Helianthus annuus 向日葵

![](/img/boxing_example.jpeg)

## Chatbot Demo Screenshots

* GIF demo

<img src="/img/bot.gif" width="400" />


* User uploads photo

![](https://i.imgur.com/nbrjfSH.png)

* Feeling lucky! for random results

![](https://i.imgur.com/FgCWr6o.png)

* Telegram app screenshot

![](https://i.imgur.com/fmMHAlo.png)

## Contributors
|Contributor|Contribute|
|:-:|:-:|
|Franklyn Chen |Help building up this project, putting in huge effort fine-tuning the model and wrote the boxing script|
|ITRI|Providing ideas, supports and the dataset|
