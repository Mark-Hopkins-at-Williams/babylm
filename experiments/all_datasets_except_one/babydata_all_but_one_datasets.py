from datasets import Dataset, DatasetDict


def read_lines(filenames, bos_tok=None, eos_tok=None):
    """Creates a generator over the lines of the provided files.
    
    If bos_tok is not None, then this token will be prepended to each line.
    If eos_tok is not None, this this token will be appended to each line.

    Each item of the generator will be a dictionary that maps the key 'text'
    to the next unread line.

    Parameters
    ----------
    filenames : list[str]
        a list of filenames
    bos_tok : str
        a 'beginning of sentence' token (or None if no BOS token is desired)
    eos_tok : str    
        an 'end of sentence' token (or None if no EOS token is desired)
    """

    for filename in filenames:
        with open(filename) as reader:
            for line in reader:
                line = line.strip()                
                if len(line) > 0:
                    if bos_tok is not None:
                        line = f"{bos_tok} {line}"
                    if eos_tok is not None:
                        line = f"{line} {eos_tok}"
                    yield {'text': line}            


def create_dataset(filenames, bos_tok, eos_tok):
    """Creates a transformers Dataset that iterates over the lines of the provided files."""
    return Dataset.from_generator(lambda: read_lines(filenames, bos_tok, eos_tok))


def create_dataset_dict(train_files, valid_files, test_files, bos_tok, eos_tok):
    """Creates a transformers DatasetDict."""
    result = DatasetDict()
    result['train'] = create_dataset(train_files, bos_tok, eos_tok)
    result['valid'] = create_dataset(valid_files, bos_tok, eos_tok)
    result['test'] = create_dataset(test_files, bos_tok, eos_tok)
    return result


def strict_small_leave_one_out(leave_out_dataset, bos_tok=None, eos_tok=None):
    """Creates a transformers DatasetDict for the strict-small babylm corpus.
    Ommits the dataset named leave_out_dataset.train/dev/test from the corpus."""
    
    corpora = ['aochildes', 'bnc_spoken', 'open_subtitles',
               'children_stories', 'cbt', 'gutenberg', 
               'qed', 'simple_wikipedia', 'switchboard', 'wikipedia']
    corpora = [i for i in corpora if i != leave_out_dataset]
    print(corpora)
    
    train_corpora = [f'../babylm_data/babylm_10M/{corpus}.train' for corpus in corpora]
    dev_corpora = [f'../babylm_data/babylm_dev/{corpus}.dev' for corpus in corpora]
    test_corpora = [f'../babylm_data/babylm_test/{corpus}.test' for corpus in corpora]
    return create_dataset_dict(train_corpora, dev_corpora, test_corpora, bos_tok, eos_tok)