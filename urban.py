# https://datamuni.com/@shivanandroy/transformers-generating-arxiv-ds-title-from-abstracts

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from simpletransformers.config.model_args import Seq2SeqArgs
from simpletransformers.seq2seq import Seq2SeqModel
# from tabulate import tabulate
from tabulate import tabulate

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.WARNING)

pd.set_option('display.max_colwidth', None)

DATA_FILE = 'words.json'


def get_script_arguments():
    parser = ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--resume_from', default='outputs')
    return parser.parse_args()


def get_metadata():
    with open(DATA_FILE, 'r') as f:
        for line in f:
            yield line


def main():
    args = get_script_arguments()
    words = []
    definitions = []
    metadata = get_metadata()
    count = 0
    for i, d in enumerate(metadata):
        if args.debug and i > 10_000:
            break
        d_dict = json.loads(d)
        definition = d_dict['definition']
        if d_dict['thumbs_up'] > d_dict['thumbs_down'] and d_dict['thumbs_up'] > 3:
            definitions.append(definition)
            words.append(d_dict['word'])
    print(count)

    data = pd.DataFrame({'word': words, 'definition': definitions})
    print(data.head())
    data = data[['word', 'definition']]
    data.columns = ['target_text', 'input_text']
    data['prefix'] = 'summarize'  # TODO: useful?
    data.drop_duplicates(inplace=True)

    # splitting the data into training and test dataset
    eval_df = data.sample(frac=0.05, random_state=101)
    train_df = data.drop(eval_df.index)
    print('TRAIN', train_df.shape, 'EVAL', eval_df.shape)

    model_args = Seq2SeqArgs()
    if args.debug:
        model_args.overwrite_output_dir = True
        model_args.num_train_epochs = 1
    else:
        model_args.num_train_epochs = 10
    model_args.evaluate_generated_text = True
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.thread_count = 1
    model_args.evaluate_during_training = False
    model_args.evaluate_during_training_verbose = False
    model_args.evaluate_during_training_steps = int(1e9)  # we don't want.
    model_args.early_stopping = False  # we don't want.
    model_args.save_steps = 50_000
    model_args.train_batch_size = 16

    if Path(args.resume_from).exists():
        print('We found a previous checkpoint...')
        name = args.resume_from
    else:
        print('We will train from the Facebook pretrained model...')
        name = 'facebook/bart-base'
    # Initialize model
    # noinspection PyArgumentEqualDefault
    model = Seq2SeqModel(
        encoder_decoder_type='bart',
        encoder_decoder_name=name,
        args=model_args,
        use_cuda=True,
    )

    if not args.eval_only:
        model.train_model(train_df, eval_data=eval_df)

    # # Evaluate the model
    results = model.eval_model(eval_df)
    print(results)

    target_text = list(eval_df['target_text'])
    input_text = list(eval_df['input_text'])
    predicted_text = model.predict(list(eval_df['input_text']))

    output_2 = []
    for t, i, p in zip(target_text, input_text, predicted_text):
        output_2.append({'target': t, 'input': i, 'predicted': p})

    output = {
        'target_text': target_text,
        'input_text': input_text,
        'predicted_text': predicted_text
    }

    with open('eval.json', 'w') as w:
        json.dump(obj=output, fp=w, ensure_ascii=False, indent=2)
    with open('eval2.json', 'w') as w:
        json.dump(obj=output_2, fp=w, ensure_ascii=False, indent=2)
    if args.debug:
        print(tabulate({
            'target_text': target_text,
            'input_text': [a[0:80] + '...' for a in input_text],
            'predicted_text': predicted_text
        }, headers='keys'))


if __name__ == '__main__':
    main()
