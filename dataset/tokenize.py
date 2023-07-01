import re

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
        return self.seq2seq_tokenize(row)

class TokenizeMapWrapper(TokenizeMapWrapper):
    def __init__(self, tokenizer, feature, max_token=4096, option=None):
        if option is None:
            option = {
                'max_length': max_token,
                'truncation': True,
            }

        self.max_token = option['max_new_tokens']
        self.option = option
        self.feature = feature
        self.tokenizer = tokenizer

    def __call__(self, row):
        total_text = row[self.feature]
        if len(re.findall('\nSummary: \n', total_text)) == 1:
            text, summary = total_text.split('Summary: \n')
            summary = '\nSummary: \n' + summary
        else:
            print('warning: more than two summary exists')
            text_split = total_text.split('Summary: \n')
            text = text_split[0]
            summary = '\nSummary: \n'.join(text_split[1:])
        
        tokenized_text = self.tokenizer(text, **self.option)
        tokenized_summary = self.tokenizer(summary, **self.option)
        tokenized_total_text = dict()
        if len(tokenized_text['input_ids']) + len(tokenized_summary['input_ids']) <= self.max_token:
            for key in tokenized_text:
                tokenized_total_text[key] = tokenized_text[key] + tokenized_summary[key]
                if len(tokenized_total_text[key]) < self.max_token:
                    tokenized_total_text[key] = (tokenized_total_text[key] 
                                                 + [1] * (self.max_token - len(tokenized_total_text[key]))
                    )
        else:
            for key in tokenized_text:
                tokenized_total_text[key] = (tokenized_text[key][:- len(tokenized_summary['input_ids'])] 
                                             + tokenized_summary[key]
                )

        return tokenized_total_text
