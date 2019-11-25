import torchvision.transforms as transforms
import torch
import torch.nn as nn
from Image_Description.Model import EncoderModel, DecoderModel
from Train_Attention import train_attention_model
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchtext.vocab import Vocab
from collections import Counter
from pycocotools.coco import COCO
from Image_Description.DataLoader import CocoDataset
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt
import nltk

device = torch.device('cuda')

class Main():
    def __init__(self):

        self.json = 'annotations/captions_val2017.json'
        self.minCount = 5
        self.imgDir = 'val2017' #coco/images'
        self.batch_size = 32
        self.embed_size = 128
        self.hidden_size = 128
        self.num_layers = 1
        self.learning_rate = 0.001
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 10
        self.vocab = self.build_vocab(self.json, self.minCount)
        self.transform = self.transform()
        self.vocab_size = len(self.vocab)
        torch.manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_dataset()

        self.enc_model = EncoderModel(self.embed_size)#.to(device)
        self.dec_model = DecoderModel(self.embed_size , self.hidden_size, self.vocab_size, self.num_layers)#.to(device)

        self.params = list(self.dec_model.parameters()) + list(self.enc_model.linear.parameters()) + list(self.enc_model.bn.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.learning_rate)

    def build_vocab(self, json, threshold):
        coco = COCO(json)
        counter = Counter()
        ids = coco.anns.keys()

        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokenized_caption = nltk.tokenize.word_tokenize((caption.lower()))
            counter.update(tokenized_caption)


        words = [word for word, count in counter.items() if count >= threshold]

        vocab = Vocab(Counter(words), specials=['<pad>', '<start>', '<end>', '<unk>'])

        return vocab

    def transform(self):
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
        ])

        return transform

    def collate_fn(self, data):
        """Creates mini-batch tensors from the list of tuples (image, caption).
        """

        # Sort a data list by caption length (descending order).
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        # Merge captions (from tuple of 1D tensor to 2D tensor).
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()

        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return images, targets, lengths

    def get_dataset(self):

        coco = CocoDataset(imgDir=self.imgDir, annDir=self.json, vocab=self.vocab, transform=self.transform)

        total_length = len(coco)
        train_length = int(0.8 * total_length) + 1
        val_length = int(0.1 * total_length)
        test_length = int(0.1 * total_length)

        train_dataset, val_dataset, test_dataset = random_split(coco, [train_length, val_length, test_length])

        return train_dataset, val_dataset, test_dataset


    def train(self):

        train_data_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True,collate_fn=self.collate_fn)
        total_step = len(train_data_loader)

        for epoch in range(self.num_epochs):
            for i, (images, captions, lengths) in enumerate(train_data_loader):
                self.enc_model.train()
                self.dec_model.train()
                # images = images.to(device)
                # captions = captions.to(device)

                features = self.enc_model(images)
                outputs = self.dec_model(features, captions, lengths)

                labels = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                loss = self.criterion(outputs, labels)
                self.enc_model.zero_grad()
                self.dec_model.zero_grad()

                loss.backward()
                self.optimizer.step()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_epochs, i+1, total_step, loss.item()))                
                if i%300 == 0:
                    val_loss = self.validate()
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch+1, self.num_epochs, i+1, total_step, loss.item(), val_loss, np.exp(val_loss)))
                    torch.save(self.enc_model , 'enc_model.pt')
                    torch.save(self.dec_model , 'dec_model.pt')

    def train_attention(self):

        train_data_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                                                        shuffle=True, collate_fn=self.collate_fn)
        val_data_loader = torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=self.batch_size,
                                                      shuffle=True, collate_fn=self.collate_fn)
        train_attention_model(self.vocab_size, train_data_loader, val_data_loader)


    def validate(self):

        val_data_loader = torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=True,collate_fn=self.collate_fn)
        total_steps = len(val_data_loader)

        self.enc_model.eval()
        self.dec_model.eval()
        val_loss = 0.0
        for i, (images, captions, lengths) in enumerate(val_data_loader):
            # images = images.to(device)
            # captions = captions.to(device)

            features = self.enc_model(images)
            outputs = self.dec_model(features, captions, lengths)

            labels = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            val_loss += self.criterion(outputs, labels).item()
            print("I in val : ",i)

        return val_loss/total_steps
      
    def evaluate(self):

        val_data_loader = torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=2, shuffle=True,collate_fn=self.collate_fn)

        enc_model_saved = torch.load('enc_model.pt',map_location=torch.device('cuda'))
        dec_model_saved = torch.load('dec_model.pt',map_location=torch.device('cuda'))

        enc_model_saved.eval()
        dec_model_saved.eval()
        valiter = iter(val_data_loader)
        images, captions, lengths = valiter.next()
        img = images[0]
        img = img.numpy()
        print(captions)
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.show()
        # images = images.to(device)
        # captions = captions.to(device)
        for t in captions:
            for val in t:
                print(self.vocab.itos[val],end=" ")  
            break
        print()
        features = enc_model_saved(images)
        sampled_ids = dec_model_saved.sample(features)
        sampled_ids = sampled_ids[0].cpu().numpy() 
        sampled_caption = []
        for word_id in sampled_ids:
            word = self.vocab.itos[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        print(sentence)

if __name__ == '__main__':

    main = Main()
    # main.train_attention()
   # main.evaluate()
    main.train()
