import argparse
import torch
import os

from fairseq.data import Dictionary
from torch.nn import Embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True,
                        help="path for source files")
    parser.add_argument("--lang", type=str, default='si',
                        help="language")
    parser.add_argument("--output", type=str, default='merged_embedding.txt',
                        help="output file name")
    parser.add_argument("--pos_file", type=str, default='dict.tagged.txt',
                        help="tags file name")
    parser.add_argument("--lang_dim", type=int, default=384,
                        help="input dim for lang")
    parser.add_argument("--pos_dim", type=int, default=128,
                        help="pos-dim")

    args = parser.parse_args()

    def load_dictionary(path, lang):
        return Dictionary.load(os.path.join(path, 'dict.{}.txt'.format(lang)))

    def build_embedding(dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        emb = normalize_embedding(emb, padding_idx)
        return emb

    def normalize_embedding(embedding, padding_idx):
        torch.nn.init.normal_(embedding.weight, mean=0, std=embedding.embedding_dim ** -0.5)
        torch.nn.init.constant_(embedding.weight[padding_idx], 0)
        return embedding

    def concat_embeddings(embedding_1, embedding_2):
        if (embedding_1.num_embeddings != embedding_2.num_embeddings):
            raise ValueError("Number of embeddings (num_embeddings) should match")

        embedded_weight = torch.cat((embedding_1.weight, embedding_2.weight), dim=1)
        embedding_concat = Embedding.from_pretrained(embedded_weight)
        return embedding_concat

    def merge_embeddings(embedding_1, embedding_2, mapping):

        weights = []
        for i in range(embedding_1.num_embeddings):  # dict file eke embeddig walata loop kala
            # Map the correct tag by mapping given in 'tags' list
            input_1 = torch.LongTensor([i])  # for embedding_1
            input_2 = torch.LongTensor([mapping[i]])  # for embedding_2 (tags 10n ekak select wenawa)

            weights.append(torch.cat((embedding_1(input_1), embedding_2(input_2)), dim=1))

        cat_weights = torch.cat(weights, dim=0)
        final_embedding = Embedding.from_pretrained(cat_weights)

        return final_embedding

    def write_embedding(embedding, dictionary, output_name="merged_embedding.txt", path="data-bin/en_si_bpe5000/"):
        embedding_file = open(os.path.join(path, output_name), 'w')
        embedding_file.write(str(embedding.num_embeddings) + " " + str(embedding.embedding_dim) + "\n")

        for index, token in enumerate(dictionary):
            if index >= len(dictionary):  # enumerating Dictionary does not stop, need to break
                break
            embedding_file.write(token + " ")
            for i in embedding(torch.LongTensor([dictionary.index(token)])).detach().numpy()[0]:
                embedding_file.write(str(i) + " ")
            embedding_file.write("\n")

    def create_pos_tags(file_location):
        data = open(file_location, encoding='utf8').readlines()

        lines = []
        tags = []

        for line in data:
            lines.append(line)
            if (line.split()[1] not in tags):
                tags.append(line.split()[1])

        dict = {}
        for i in range(len(tags)):
            dict[tags[i]] = i

        l = len(dict)  #length of the current dictionary
        dict['pad'] = len(dict) 
        #dict['UNK'] = len(dict)
               

        tag_mapping = [dict['UNK'], dict['pad'], dict['UNK'], dict['UNK']]
        for line in data:
            tag_mapping.append(dict[line.split()[1]])

        return tag_mapping, len(dict)

    lang_dict = load_dictionary(args.path, args.lang)

    lang_embedding = build_embedding(lang_dict, args.lang_dim)
    tag_mapping, num_tags = create_pos_tags(args.pos_file)
    ##
    print(len(lang_dict), len(tag_mapping), num_tags)
    pos_embedding = Embedding(num_tags, args.pos_dim)
    pos_embedding = normalize_embedding(pos_embedding, num_tags - 1)

    merged_embedding = merge_embeddings(lang_embedding, pos_embedding, tag_mapping)

    # input = torch.LongTensor([1])
    # print("Lang:", lang_embedding(input))
    # print("Pos:", pos_embedding(torch.LongTensor([tag_mapping[0]])))
    # print("Full:", merged_embedding(input))
    # print("Dimension: ", merged_embedding.num_embeddings, merged_embedding.embedding_dim)


    write_embedding(merged_embedding, lang_dict, args.output)


if __name__ == "__main__":
    main()


# !python scripts/embedding_generator.py\
#     --path "data-bin/si_en_bpe5000/" \
#     --lang "new" \
#     --pos_file "dict.tagged.txt" \
#     --output "merged_embedding.txt" \
#     --lang_dim $lang_embed_dim --pos_dim $pos_embed_dim
