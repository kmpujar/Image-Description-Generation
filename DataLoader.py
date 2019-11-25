import torch
import torch.utils.data as data
import os
import nltk
from PIL import Image
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):

    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, imgDir, annDir, vocab, transform):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = imgDir
        self.coco = COCO(annDir)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab.stoi['<start>'])
        caption.extend([vocab.stoi[token] for token in tokens])
        caption.append(vocab.stoi['<end>'])
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)
