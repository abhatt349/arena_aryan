# %% 

import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2") #   type: ignore

# %%

def test(word):
    print(word + ':', '-'*(25-len(word)), tokenizer.encode(word))
    
# %% 

print(tokenizer.encode('the happy monkey'))
print(tokenizer.encode('the'))
print(tokenizer.encode(' happy'))
print(tokenizer.encode(' monkey'))
print(tokenizer.encode('happy monkey'))

# %%

print('Testing tokenizer.encode')
test('stew')
test('steward')
test('stewardess')
test('stewardesses')
test('ess')
test('esses')



print('|'.join([tokenizer.decode(x) for x in range(10000,10020)]))

# %%

print(tokenizer.tokenize('happy as a cute lil clam'))
print(tokenizer('happy as a cute lil clam'))
print(tokenizer.encode('happy as a cute lil clam'))


# %%

print('|'.join([c for c in tokenizer.all_special_tokens]))

# %%




