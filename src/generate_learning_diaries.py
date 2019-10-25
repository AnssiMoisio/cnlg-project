#!/usr/bin/env python3
import fire
import json
import os
import numpy as np
import tensorflow as tf
import string
import model, sample, encoder
import time
import tensorflow_hub as hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

# ld_file = os.path.join('.','learning_diary.txt')
# abstract_file = os.path.join('.','abstract.txt')

graph = tf.Graph()

def create_embed(texts):
    """
    embeddings for each text
    """
    with tf.Session(graph = graph) as session:
        embed = hub.Module(module_url)
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        text_embeddings = session.run(embed(texts))

    return text_embeddings


def similarity(v1, v2):
    """
    similarity between two vector embeddings
    """
    return np.inner(v1, v2)


def select_best_learning_diary(aesthetic_values):
    """
    select the best learning diary based on aesthetic values
    """
    linear_combination = np.zeros((len(aesthetic_values[0])))
    for array in aesthetic_values:
        for i in range(len( array)):
            linear_combination[i] += array[i]

    return np.argmax(linear_combination)


def interact_model(
    model_name='774M',
    seed=None,
    nsamples=2,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=40,
    top_p=1,
    # script needs to be run from project root folder (not src)
    models_dir=os.path.join('.','models'), 
    model_text=os.path.join('.','model_text.txt'),
    learning_diary_file=os.path.join('.','learning_diary_words.txt'),
    output_dir=os.path.join('.','generated_texts'),
):
    """
    Run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))

    with open(model_text, "r") as f:
        raw_text = f.read()

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    
    with tf.Session(graph = graph) as sess:
        
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)
        
        context_tokens = enc.encode(raw_text)
        generated = 0
        texts_print = "=" * 40 + " Model text " + "=" * 40 + "\n" + raw_text + "\n" + "=" * 80 # input text
        print(texts_print)
        texts = []
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                texts.append(text)
                new_text = "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n" + text + "\n\n"
                print(new_text)
                texts_print += new_text
        print("=" * 80)
        texts_print += "=" * 80

        ## save generated learning diaries to file
        timestr = time.strftime("%Y%m%d-%H%M%S") # time stamp to file name
        file_name = "output_texts_print" + timestr + ".txt"
        with open(os.path.join(output_dir, file_name), "w") as f:
            f.write(texts_print)

        # sentence embeddings for the texts
        text_embeddings = create_embed(texts)
        model_embed = create_embed([raw_text])

        # similarity to model text (first aesthetic)
        sims_to_model = []
        for vec in text_embeddings:
            sims_to_model.append(similarity(model_embed[0], vec))
        
        # similarity to a learning diary example (second aesthetic)
        with open(learning_diary_file, "r") as f:
            learning_diary_words = f.read()
        ld_embed = create_embed([learning_diary_words])
        
        sims_to_ld = []
        for vec in text_embeddings:
            sims_to_ld.append(similarity(vec, ld_embed[0]))

        aesthetic_values = [sims_to_model, sims_to_ld]
        print("The best diary is sample number", 1+select_best_learning_diary(aesthetic_values))
        print("Aesthetic values:\n", aesthetic_values)

        ## Reference similarity values
        # with open(ld_file, "r") as f:
        #     ld = f.read()
        # v = create_embed([ld])
        # with open(abstract_file, "r") as f:
        #     a = f.read()
        # a_embed = create_embed([a])
        # sim_to_ld = similarity(v, ld_embed) 
        # sim_to_a = similarity(v, a_embed)
        # print("Reference similarity values: ", sim_to_ld, sim_to_a)
        

if __name__ == '__main__':
    fire.Fire(interact_model)
