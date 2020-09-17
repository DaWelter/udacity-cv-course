import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

def indexgrid2d(shape):
    # Generate a tensor with grid indices for a tensor of size shape.
    # Shape must be two-dimensional.
    # Output is a 3d tensor. It can be split like this
    # row_indices, col_indices = indexgrid2d(...
    # Now row indices, and col_indices both have the input shape.
    # Their values indicate row and column indices, respectively.
    # E.g.
    #     indexgrid2d((5,2))
    #     tensor([[[ 0,  0],
    #          [ 1,  1],
    #          [ 2,  2],
    #          [ 3,  3],
    #          [ 4,  4]],

    #         [[ 0,  1],
    #          [ 0,  1],
    #          [ 0,  1],
    #          [ 0,  1],
    #          [ 0,  1]]], dtype=torch.int32)
    r = torch.arange(shape[0], dtype=torch.long)[:,None]
    c = torch.arange(shape[1], dtype=torch.long)[None,:]
    c = c.repeat(shape[0], 1)
    r = r.repeat(1, shape[1])
    return torch.cat((r[None,...],c[None,...]), dim=0)


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.beam_search_k = 20
    
    def forward(self, features, captions):
        captions = self.embed(captions[:,:-1])
        features = features.unsqueeze(1)
        x = torch.cat([features, captions], dim=1)
        y, state = self.lstm(x)
        y = self.fc(y)
        return y
        
    # General sampling utilities
    
    def generate_words(self, lstm_outputs, k):
        # This function samples k words from each item in the batch.
        # lstm_outputs: (batch_size, 1, hidden_size) Tensor
        # Outputs:
        #    Tensor with word indices of size (batch_size, k).
        #    Tensor of same size with corresponding logits.
        assert lstm_outputs.size(1) == 1
        lstm_outputs = lstm_outputs[:,0,:]
        assert len(lstm_outputs.shape)==2
        # First compute the probabilities
        scores = self.fc(lstm_outputs)
        probs = F.softmax(scores, dim = 1)
        # Sample
        indices = torch.multinomial(probs, num_samples=k)
        scores_of_indices = torch.gather(scores, 1, indices)
        return indices, scores_of_indices


    def embed_word_indices(self, indices):
        x = self.embed(indices)
        # Adds the sequence dimension from which only one entry is needed.
        return x[:,None,:]
    
    # Beam search, taking inspiration from the paper "Show and Tell: A Neural Image Caption Generator"
    # Plenty of blog posts explaining the technique. This one is not too bad
    # https://hackernoon.com/beam-search-a-search-strategy-5d92fb7817f
    
    def beam_search_step(self, sentences, sentence_scores, prev_word_vecs, lstm_state):
        # Performs the word selection and updating of sentences for one step of the
        # Beam Search algorithm. I use the batch dimension to store the beam_search_k different
        # sentence/word candidates.
        # Inputs:
        #   sentences: List of list of word indices. Sentences generated in the previous steps
        #   sentence_scores: something like the log probabilities of the sentences.
        #                    I'm using the sum of raw word scores directly.
        #   prev_word_vecs: Word vectors from the last step. That is a batch of beam_search_k
        #                   number of words, except for in the first step where the batch size
        #                   is 1.
        #   lstm_state: The last state of the lstm network.
        
        y, lstm_state = self.lstm(prev_word_vecs, lstm_state)
        
        # Rows in the result below run over the batch, i.e. over the input sentences/words.
        # Columns run over different samples generated from each row.
        indices, scores = self.generate_words(y, self.beam_search_k)
        
        # Compute total sentence score. Broadcast previous sentence scores.
        # Want to replicate prev_word_scores across columns.
        candidate_scores = scores + sentence_scores[:,None]
        
        # Now we can grab the best performing pairs of previous sentence and new word.
        topk_scores, topk_indices = torch.topk(
            candidate_scores.reshape(-1), self.beam_search_k)
        topk_row, topk_col = indexgrid2d(indices.shape).reshape(2,-1)[:,topk_indices]
        
        # The next round has to use the states from which the
        # top candidates were selected.
        h, c = lstm_state
        assert h.dim() == 3
        assert h.size(1) == indices.size(0)
        h = h[:,topk_row,:]
        c = c[:,topk_row,:]
        
        next_words = indices.reshape(-1)[topk_indices]
        next_word_vecs = self.embed_word_indices(next_words)
        
        # Similar to the state manipulation above, we must continue
        # with the right input sentence. Here the new sentences
        # are assembled from the old-sentence, new-word pairs.
        new_sentences = []
        for i,w in zip(topk_row, next_words):
            new_sentences.append(sentences[i] + [int(w.cpu())])
        
        return new_sentences, topk_scores, next_word_vecs, (h,c)

    
    def beam_search(self, inputs, max_len):
        # Generates the complete sentence candidates including final scores.
        # Simply calls the step function in a recurrent fashion.
        with torch.no_grad():
            sentences = [[]]
            sentence_scores = torch.zeros((1,), device=inputs.device)
            word_vecs = inputs
            lstm_state = None
            for _ in range(max_len):
                sentences, sentence_scores, word_vecs, lstm_state = self.beam_search_step(
                    sentences, sentence_scores, word_vecs, lstm_state)
            return sentences, sentence_scores
    
    
    def naive_sampling(self, word_vec, max_len):
        # Aka greedy search. Outputs the sampled sentence.
        with torch.no_grad():
            output = []
            hc = None
            for _ in range(max_len):
                y, hc = self.lstm(word_vec, hc)
                word_index, _ = self.generate_words(y, 1)
                word_vec = self.embed_word_indices(word_index[0])
                output.append(int(word_index[0,0].cpu()))
            return output
    
    
    def sample(self, inputs, states=None, max_len=20, beam_search=True):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        if beam_search:
            sentences, _ = self.beam_search(inputs, max_len)
            # Return the best scoring one.
            return sentences[0]
        else:
            return self.naive_sampling(inputs, max_len)