# https://datamuni.com/@shivanandroy/transformers-generating-arxiv-ds-title-from-abstracts

import json
import logging
import sys

import pandas as pd
from simpletransformers.config.model_args import Seq2SeqArgs
from simpletransformers.seq2seq import Seq2SeqModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.WARNING)

pd.set_option('display.max_colwidth', None)

DATA_FILE = 'words.json'


def get_metadata():
    with open(DATA_FILE, 'r') as f:
        for line in f:
            yield line


def main():
    debug = len(sys.argv) > 1
    words = []
    definitions = []
    metadata = get_metadata()
    count = 0
    for i, d in enumerate(metadata):
        if debug and i > 10_000:
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

    # splitting the data into training and test dataset
    eval_df = data.sample(frac=0.05, random_state=101)
    train_df = data.drop(eval_df.index)
    print('TRAIN', train_df.shape, 'EVAL', eval_df.shape)

    model_args = Seq2SeqArgs()
    if debug:
        model_args.overwrite_output_dir = True
        model_args.num_train_epochs = 1
    else:
        model_args.num_train_epochs = 1
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

    # Initialize model
    # noinspection PyArgumentEqualDefault
    model = Seq2SeqModel(
        encoder_decoder_type='bart',
        encoder_decoder_name='facebook/bart-base',
        args=model_args,
        use_cuda=True,
    )

    model.train_model(train_df, eval_data=eval_df)

    # # Evaluate the model
    results = model.eval_model(eval_df)

    random_num = 350
    actual_title = eval_df.iloc[random_num]['target_text']
    actual_abstract = [eval_df.iloc[random_num]['input_text']]
    predicted_title = model.predict(actual_abstract)

    print(f'Actual Word: {actual_title}')
    print(f'Predicted Word: {predicted_title}')
    print(f'Actual Definition: {actual_abstract}')

    # Use the model for prediction
    print(
        model.predict(
            [
                "Tyson is a Cyclops, a son of Poseidon, and Percy Jackson’s half brother. He is the current general of the Cyclopes army."
            ]
        )
    )

    # model_args = T5Args()
    # model_args.max_seq_length = 96
    # model_args.train_batch_size = 20
    # model_args.eval_batch_size = 20
    # model_args.num_train_epochs = 1
    # model_args.evaluate_during_training = True
    # model_args.evaluate_during_training_steps = 30000
    # model_args.use_multiprocessing = False
    # model_args.fp16 = False
    # model_args.save_steps = -1
    # model_args.save_eval_checkpoints = False
    # model_args.no_cache = True
    # model_args.reprocess_input_data = True
    # model_args.overwrite_output_dir = True
    # model_args.preprocess_inputs = False
    # model_args.num_return_sequences = 1
    # model_args.use_multiprocessing_for_evaluation = True
    # model_args.process_count = 2

    # Create T5 Model
    # model = T5Model('mt5', 't5-small', args=model_args)

    # Train T5 Model on new task
    # model.train_model(train_df, eval_data=eval_df)
    #
    # # Evaluate T5 Model on new task
    # results = model.eval_model(eval_df.head(10))
    #
    # print(results)
    #
    # random_num = 350
    # actual_title = eval_df.iloc[random_num]['target_text']
    # actual_abstract = ['summarize: ' + eval_df.iloc[random_num]['input_text']]
    # predicted_title = model.predict(actual_abstract)
    #
    # print(f'Actual Title: {actual_title}')
    # print(f'Predicted Title: {predicted_title}')
    # print(f'Actual Abstract: {actual_abstract}')


if __name__ == '__main__':
    main()
