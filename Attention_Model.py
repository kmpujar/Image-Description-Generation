import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionEncoder(nn.Module):

    def __init__(self, encoded_image=14):
        super(AttentionEncoder, self).__init__()

        self.enc_image_size = encoded_image
        self.resnet = torchvision.models.resnet101(pretrained=True)

        self.modules = list(self.resnet.children())[:-2]
        self.resnet = nn.Sequential(*self.modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image, encoded_image))


    def forward(self, images):

        out = self.resnet(images)
        # Adaptive pooling of layers
        out = self.adaptive_pool(out)

        #Swapping dimensions to get L features ( 2048 in the last layer - (batch_size, 2048, encoded_image_size, encoded_image_size) -> (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.permute(0, 2, 3, 1)
        return out


class AttentionNetwork(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(AttentionNetwork, self).__init__()

        #Encoder Attention Layer
        self.encoder_attention = nn.Linear(encoder_dim, attention_dim)

        #Decoder Attention Layer
        self.decoder_attention = nn.Linear(decoder_dim, attention_dim)

        #Combining both layers
        self.full_attention = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):

        att1 = self.encoder_attention(encoder_out)
        att2 = self.decoder_attention(decoder_hidden)

        # Dimension of combined layer is encoder attention + decoder attention layer
        attention = self.full_attention(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)

        # Attention Weights
        alpha = self.softmax(attention)

        # Weighted encoding layers
        att_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return att_weighted_encoding, alpha

class AttentionDecoder(nn.Module):

    def __init__(self, attention_dim, embedding_dim, decoder_dim, vocab_size, dropout, encoder_dim=2048):
        super(AttentionDecoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dropout = dropout


        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTMCell(embedding_dim + encoder_dim, decoder_dim)
        self.h0 = nn.Linear(encoder_dim, decoder_dim)
        self.c0 = nn.Linear(encoder_dim, decoder_dim)
        self.gate = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(decoder_dim, vocab_size)
        self.attention = AttentionNetwork(encoder_dim, decoder_dim, attention_dim)
        self.init_weights()

    def init_weights(self):

        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_states(self, encoder_out):

        # Initializing first hidden state with average of encoder output -> According to paper
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.h0(mean_encoder_out)
        c = self.c0(mean_encoder_out)

        return h,c

    def forward(self, encoder_out, captions, caption_len):

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Reshaping according to batch size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sorting according to descending lengths -> Used for dynamic batching (explained afterwards)
        # print(caption_len.size())
        caption_len, index = caption_len.sort(dim=0, descending=True)
        encoder_out = encoder_out[index]
        captions = captions[index]

        embedding = self.embedding(captions)
        h, c = self.init_hidden_states(encoder_out)

        # Using length - 1 to not include <end>
        decoder_length = (caption_len - 1).tolist()
        # print("Decoder length : ",decoder_length)
        pred = torch.zeros(batch_size, max(decoder_length), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decoder_length), num_pixels).to(device)
        # print("Pred Size : ",pred.size())
        # print("Alpha Size : ",alphas.size())
        # Dynamic batching used here. We use lstm cell which runs over only one time step at once. We only go through samples which doesn't have <pad> at that timestep.
        for i in range(max(decoder_length)):
            batch_size_i = sum([l > i for l in  decoder_length])

            #Getting attention weights and weighted encoded values
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_i], h[:batch_size_i])


            #Sigmoid gating -> Paper says this gives better results
            gate = self.sigmoid(self.gate(h[:batch_size_i]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            #Passing embedding layer output and attention_weighted encoding output concatenated together to next layer of lstm
            h, c = self.lstm(torch.cat([embedding[:batch_size_i, i, :], attention_weighted_encoding], dim=1), (h[:batch_size_i], c[:batch_size_i]))


            #Predicted output
            preds = self.linear(self.dropout(h))
            pred[:batch_size_i, i, :] = preds
            alphas[:batch_size_i, i, :] = alpha


        return pred, captions, decoder_length, alphas, index

