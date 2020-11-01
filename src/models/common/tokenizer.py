from janome.tokenizer import Tokenizer


class JanomeTokenizer:
    def __init__(self, part_of_speech=['名詞', '形容詞']):
        self.j = Tokenizer()
        self.part_of_speech = part_of_speech
    
    def __call__(self, doc):
        list_ = [token.surface for token in self.j.tokenize(doc) 
                 if token.part_of_speech.split(',')[0] in self.part_of_speech]
        return list_
    
    
# class MecabTokenizer:
#     def __init__(self, part_of_speech=['名詞', '形容詞']):
#         self.m = Tokenizer()
#         self.part_of_speech = part_of_speech
    
#     def __call__(self, doc):
#         pass
    