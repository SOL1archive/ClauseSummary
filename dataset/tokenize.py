class TokenizeMapWrapper:
    def __init__(self, tokenizer, feature, option=None):
        if option is None:
            option = {
                'max_length': 512,
                'truncation': True,
                'padding': 'max_length',
            }
        
        self.feature = feature
        self.tokenizer = tokenizer

    def __call__(self, row):
        return self.tokenizer(row[self.feature], **self.option)

    def __repr__(self):
        return f'{self.__class__.__name__}(tokenizer={self.tokenizer})'

class Seq2SeqTokenizeMapWrapper(TokenizeMapWrapper):
    def __init__(self, tokenizer, feature, target, option=None):
        super().__init__(tokenizer, feature, option)
        self.target = target

    def seq2seq_tokenize(self, row):
        form_embeddings = self.tokenizer(row[self.feature], **self.option)
        with self.tokenizer.as_target_tokenizer():
            correct_form_embeddings = self.tokenizer(row[self.target], **self.option)

        return {
            'input_ids': form_embeddings['input_ids'],
            'attention_mask': form_embeddings['attention_mask'],
            'labels': correct_form_embeddings['input_ids'],
        }

    def __call__(self, row):
        return self.seq2seq_tokenize(row, self.tokenizer)
