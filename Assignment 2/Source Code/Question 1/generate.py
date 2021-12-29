###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./models/model-Transformer-tied.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--ngram', type=int, default=8,
                    help='ngram value')                    
args = parser.parse_args()

# Print Model Checkpoint for logging
print("\n ===============================")
print("Model checkpoint:", args.checkpoint)

# Configure filename of generated text file
generated_text_filename = 'generated/' + args.checkpoint[9:-3] + '-' + args.outf

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

# Check if model is transformer or FNN model
is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
is_FNN_model = hasattr(model, 'model_type') and model.model_type == 'FNN'

# Initialise hidden state if RNN based architecture
if (not is_transformer_model) and (not is_FNN_model):
    hidden = model.init_hidden(1)

"""Configure input to model 
    For FNN model with ngram of example 8, the number of input words will be 7,
    and the next word will be predicted based on the first 7 words (randomly generated). 
    For Transformer or RNN architecture models, the next word will be predicted on the 
    first random word
"""
if is_FNN_model:
    input = torch.randint(ntokens, (1, args.ngram-1), dtype=torch.long).to(device)
else:
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)


"""Generation of text
    FNN Model takes in 7 words as input and generates an additinal word. 
    The output word_tensor is added to the input and the first word of 
    input is removed to maintain a size of 7 input words. For Transformer
    or RNN based models, the original generation of text is utilised.
"""
with open(generated_text_filename, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            if is_transformer_model:
                output = model(input, False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)
            elif is_FNN_model:
                output = model(input)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                # concatenate new word into the input
                input = torch.cat([input, word_tensor], 1)
                # remove first word to maintain length of 7 inputs words
                input = input[:, 1:]             
            else:
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)

            word = corpus.dictionary.idx2word[word_idx]

            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))