#!/usr/bin/env python
# coding=utf-8

# lachesis automates the segmentation of a transcript into closed captions
#
# Copyright (C) 2016-2017, Alberto Pettarin (www.albertopettarin.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
TBW
"""

from __future__ import absolute_import
from __future__ import print_function
import cPickle as pickle
import io
import os
import sys

from lachesis.downloaders import Downloader
from lachesis.elements import TokenizedSentenceSpan
from lachesis.language import Language
from lachesis.nlpwrappers import NLPEngine
import lachesis.globalfunctions as gf


def tokens_to_features(tokens, look_behind=5, look_ahead=5, max_chars_per_line=42, debug=False):
    """
    Convert a sequence of tokens into a sequence of features,
    that is, a list of dicts, each dict containing
    the features associated to the corresponding token
    in the input sequence.
    """
    feature_sequence = []

    # for debugging purposes, use this simple function
    if debug:
        for idx, token in enumerate(tokens):
            feature_sequence.append({
                "idx": idx,
                "word": token.raw,
                "ws": token.trailing_whitespace,
                "pos": token.upos_tag,
            })
        return feature_sequence

    # raw string of the word
    words = [t.raw for t in tokens]
    # POS of the word
    poses = [t.upos_tag for t in tokens]
    # bool, True if word has trailing whitespace
    wses = [t.trailing_whitespace for t in tokens]

    n = len(tokens)
    # length of the word, including trailing space, if present
    lens = [len(w) for w in words]
    lenw = [1 if ws else 0 for ws in wses]
    # cumulative length:
    # clens[i] = length of the prefix string from 0 to i (included)
    clens = [0 for i in range(n)]
    clens[0] = lens[0]
    for i in range(1, n):
        clens[i] = clens[i - 1] + lenw[i - 1] + lens[i]

    # pad look_ahead to simplify later index access
    bf_poses = [u"BOLP"] * look_behind + poses + [u"BOLP"] * look_ahead
    bf_lens = [0] * look_behind + lens + [1000] * look_ahead
    bf_clens = [0] * look_behind + clens + [1000] * look_ahead

    for idx in range(n):
        sidx = look_behind + idx
        word = words[idx]
        token_features = {
            "bias": 0,
            "idx": idx,
            "bol": (idx == 0),
            "word": word,
            "pos": poses[idx],
            "ws": wses[idx],
            "len": lens[idx],
            "upper": word[0].isupper(),
            "long": len(word) > 5,
            "clen": clens[idx] > max_chars_per_line,
        }
        for b in range(1, look_behind + 1):
            # look behind
            token_features["pos-0-%d" % b] = u"-".join(bf_poses[(sidx - b):sidx])
            token_features["clen-%d" % b] = bf_clens[sidx - b] > max_chars_per_line
        for f in range(1, look_ahead + 1):
            # look ahead
            token_features["pos-0+%d" % f] = u"-".join(bf_poses[sidx:(sidx + f)])
            token_features["clen+%d" % f] = bf_clens[sidx + f] > max_chars_per_line
        for i in range(1, min(b, f)):
            # look behind-ahead
            token_features["pos-%d+%d" % (i, i)] = u"-".join(bf_poses[(sidx - i):(sidx + i)])
        # print(token_features)
        feature_sequence.append(token_features)

    #feature_sequence[0]["pseq"] = u"BOS"
    #for idx in range(1, n):
    #    feature_sequence[idx]["pseq"] = list(feature_sequence[idx - 1])
    #for idx in range(0, n - 1):
    #    feature_sequence[idx]["nseq"] = list(feature_sequence[idx + 1])
    #feature_sequence[n - 1]["nseq"] = u"EOS"

    return feature_sequence


class CRFTrainer(object):
    """
    TBW
    """

    PARAMETERS = {
        "max_iterations": 100,
        "feature.possible_transitions": True,
    }
    """ Parameters for the trainer from pycrfsuite """

    VERBOSE = False
    """ Verbosity of the trainer """

    LABEL_FIRST = u"F"
    """ Label for a token that is the begin of a line """

    LABEL_MIDDLE = u"_"
    """ Label for a token that is not the last of a line """

    LABEL_LAST = u"L"
    """ Label for a token that is the last of a line """

    def __init__(
        self,
        language,
        nlpwrapper=u"nltk",
        downloader=u"youtube",
        parameters=PARAMETERS,
        verbose=VERBOSE
    ):
        self.language = language
        self.nlpwrapper = nlpwrapper
        self.downloader = downloader
        self.parameters = parameters
        self.verbose = verbose
        self.nlpe = NLPEngine(preload=[(self.language, self.nlpwrapper)])
        self.train_data = None
        self.trainer = None

    @property
    def train_data_cc(self):
        if self.train_data is None:
            return 0
        return len(self.train_data)

    @property
    def train_data_lines(self):
        if self.train_data is None:
            return 0
        return sum([len(cc) for cc in self.train_data])

    def _read_files(self, input_file_paths):
        def _annotated_sentence_to_lines(tokens):
            lines = []
            cl = []
            for t in tokens:
                if t.is_special:
                    if len(cl) > 0:
                        lines.append(cl)
                        cl = []
                else:
                    cl.append(t)
            if len(cl) > 0:
                lines.append(cl)
            return lines

        # NOTE in theory we could account for things like the position
        #      of a token w.r.t. the whole document or the sentence it belongs
        #      however here we make an assumption of independence of the split point
        #      from those facts
        examples = []
        for ifp in input_file_paths:
            print(u".")
            if os.path.isfile(ifp):
                doc = Downloader.read_closed_captions(ifp, {u"downloader": u"youtube"})
                self.nlpe.analyze(doc, wrapper=self.nlpwrapper)
                for sentence in doc.sentences:
                    example_lines = []
                    # print(sentence.string(eol=u"|", eos=u"").strip())
                    # sentence is a Span object
                    # sentence.elements is a list of Token objects
                    lines = _annotated_sentence_to_lines(sentence.tokens)
                    n = len(lines)


                    # NOTE: this does not work
                    #       create a single example out of the entire sentence
                    #extended_line = []
                    #extended_labels = []
                    #for l in lines:
                    #    extended_line.extend(l)
                    #    #extended_labels.extend([self.LABEL_FIRST] + [self.LABEL_MIDDLE] * (len(l) - 2) + [self.LABEL_LAST])
                    #    extended_labels.extend([self.LABEL_MIDDLE] * (len(l) - 1) + [self.LABEL_LAST])
                    #extended_features = tokens_to_features(extended_line)
                    #example = (extended_features, extended_labels, extended_line)
                    #print(u"".join(extended_labels))
                    #example_lines.append(example)


                    for i in range(n):

                        extended_line = list(lines[i])
                        current_line_end_idx = len(extended_line) - 1

                        if i + 1 < n:
                            # add tokens
                            next_line = lines[i + 1]
                            extended_line.extend(next_line[:-1])

                        extended_features = tokens_to_features(extended_line)
                        #
                        #if i == 0:
                        #    # add bos
                        #    pass
                        #if i == n - 1:
                        #    # add eos
                        #    pass
                        #
                        # set the appropriate labels
                        extended_labels = [self.LABEL_MIDDLE] * len(extended_line)
                        #extended_labels[0] = self.LABEL_FIRST
                        extended_labels[current_line_end_idx] = self.LABEL_LAST

                        # generate an example, by concatenating tokens
                        # whose string representation
                        # has length <= to max_chars_per_line
                        m = len(extended_line)
                        j = 0
                        while j < m:
                            if extended_features[j]["clen"]:
                                break
                            j += 1
                        # NOTE: commented because we need to take a substring
                        #       in the next instruction
                        # j -= 1
                        example = (extended_features[:j], extended_labels[:j], extended_line[:j])
                        # print(example)
                        example_lines.append(example)


                    examples.append(example_lines)
        return examples

    def load_data(self, obj):
        """
        TBW
        """
        if isinstance(obj, list):
            # parse the given list of files
            input_file_paths = obj
            self.train_data = self._read_files(input_file_paths)
        else:
            # try loading from pickle
            input_file_path = obj
            self.train_data = pickle.load(io.open(input_file_path, "rb"))

    def dump_data(self, dump_file_path):
        """
        TBW
        """
        pickle.dump(self.train_data, io.open(dump_file_path, "wb"))

    def train(self, model_file_path):
        """
        TBW
        """
        # create a trainer object
        import pycrfsuite
        self.trainer = pycrfsuite.Trainer(algorithm="pa", verbose=self.verbose)

        # append training data
        for sentence in self.train_data:
            # each sentence contains a list of examples
            # each example is a triplet (feature_seq, label_seq, tokens)
            for feature_seq, label_seq, tokens in sentence:
                self.trainer.append(feature_seq, label_seq)

        # do the actual training
        self.trainer.train(model_file_path)

        # return the path to the model file
        return model_file_path

    def trainer_info(self):
        if self.trainer is None:
            return None
        return self.trainer.info()


class CRFPredictor(object):
    """
    TBW
    """

    LABEL_FIRST = CRFTrainer.LABEL_FIRST
    """ Label for a token that is the begin of a line """

    LABEL_MIDDLE = CRFTrainer.LABEL_MIDDLE
    """ Label for a token that is not the last of a line """

    LABEL_LAST = CRFTrainer.LABEL_LAST
    """ Label for a token that is the last of a line """

    def __init__(self, model_file_path):
        import pycrfsuite
        self.model_file_path = model_file_path
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(self.model_file_path)

    @property
    def info(self):
        """
        TBW
        """
        if self.tagger is None:
            return None
        return self.tagger.info()

    def predict(self, obj):
        """
        ``obj`` is either a list of features or a Span containing Tokens.
        In the latter case, features will be extracted from the sequence of Tokens.

        TBW
        """
        features = None
        if isinstance(obj, list):
            features = obj
        elif isinstance(obj, Span):
            sentence = obj
            tokens = [token for token in sentence.elements if token.is_regular]
            features = tokens_to_features(tokens)
        else:
            raise TypeError(u"The obj should be either a Span (sentence) object or a list of features (dict) objects.")
        predicted_labels = self.tagger.tag(features)
        probability = self.tagger.probability(predicted_labels)
        return predicted_labels, probability


def usage(exit_code):
    """ Print usage and exit. """
    print(u"")
    print(u"Usage:")
    print(u"  $ python -m lachesis.ml.crf dump  LANGUAGE INPUT_DIR DUMP_FILE  [--small]")
    print(u"  $ python -m lachesis.ml.crf train LANGUAGE DUMP_FILE MODEL_FILE")
    print(u"  $ python -m lachesis.ml.crf test  LANGUAGE DUMP_FILE MODEL_FILE [--single]")
    print(u"")
    print(u"Options:")
    print(u"  --single : DUMP_FILE is a path to a single TTML file, not to a DUMP file created with dump")
    print(u"  --small  : only use first 10 TTML files from INPUT_DIR instead of all")
    print(u"")
    sys.exit(exit_code)


def main():
    """ Entry point. """

    def check_language(obj):
        """ Check that the given string identifies a known language. """
        language = Language.from_code(obj)
        if language is None:
            print(u"[ERRO] Unknown language code '%s'" % obj)
            usage(1)
        return language

    def check_dir(obj):
        """ Check that the given string identifies an existing directory. """
        if not os.path.isdir(obj):
            print(u"[ERRO] Directory '%s' does not exist" % obj)
            usage(1)
        return obj

    def check_file(obj):
        """ Check that the given string identifies an existing file. """
        if not os.path.isfile(obj):
            print(u"[ERRO] File '%s' does not exist" % obj)
            usage(1)
        return obj

    def command_dump(language, input_directory_path, dump_file_path, small_n=None):
        """
        Create a cPickle dump with the features and labels
        from the TTML files contained in the given input directory.
        """
        input_files = []
        for root, dirs, files in os.walk(input_directory_path):
            input_files.extend([os.path.join(root, f) for f in files if f.endswith(u".ttml")])

        input_files = sorted(input_files)
        if small_n is not None:
            small_n = min(small_n, len(input_files))
            input_files = input_files[:small_n]

        trainer = CRFTrainer(language=language)

        print(u"Parsing data...")
        trainer.load_data(input_files)
        print(u"Parsing data...done")

        print(u"Dumping data...")
        trainer.dump_data(dump_file_path)
        print(u"Dumping data... done")
        print(u"Dumped %d ccs / %d lines examples to: '%s'" % (trainer.train_data_cc, trainer.train_data_lines, dump_file_path))

    def command_train(language, dump_file_path, model_file_path):
        """
        Train a CRF model from the given dump file
        containing both features and labels.
        """
        print(u"Loading data...")
        trainer = CRFTrainer(language=language, nlpwrapper=u"nltk")
        trainer.load_data(dump_file_path)
        print(u"Loading data... done")

        print(u"Training...")
        trainer.train(model_file_path)
        print(u"Training... done")
        print(u"Built model '%s'" % model_file_path)

    def command_test(language, dump_file_path, model_file_path, single):
        """
        Test a CRF model against the given dump file
        containing both features and labels.

        The predictions are accounted for using
        the same algorithm powering the actual splitter.

        TBW
        """

        from lachesis.splitters import CRFSplitter

        ANSI_ERROR = u"\033[91m"
        ANSI_OK = u"\033[92m"
        ANSI_WARNING = u"\033[93m"
        ANSI_END = u"\033[0m"

        def check_prediction(l_predictor, l_subf, l_subl, tokens):
            """
            Call the tagger and check the prediction
            against the true value (i.e., the true label).
            """
            l_str = (u"".join([u"%s%s" % (t.raw, (u" " if t.trailing_whitespace else u"")) for t in tokens])).strip()
            l_subl_pred, l_probability = l_predictor.predict(l_subf)
            l_pred_s = u"".join(l_subl_pred)
            l_real_s = u"".join(l_subl)
            return (l_real_s, l_pred_s, l_str, l_probability)

        def evaluate_pure_labelling(example_sentences, predictor):
            """
            Evaluate the pure labelling of lines, that is,
            get the label string for a line
            (knowing that it is the whole line),
            and confront it with the ground truth label string
            (which, in particular, has form: "___...___E").
            """
            print(u"Evaluating pure labelling...")
            n_sentences = len(example_sentences)
            n_sentences_good = 0
            n_examples = 0
            n_examples_equal = 0
            n_examples_not_good_with_EOL = 0
            for example_sentence in example_sentences:
                # each sentence is a list of lines
                n_examples += len(example_sentence)
                # each line is a pair (features_seq, labels_seq)
                se_good = 1
                cl_good_equal = 0
                for features, labels, tokens in example_sentence:
                    real_s, pred_s, str_s, prob = check_prediction(predictor, features, labels, tokens)
                    #print(real_s, str_s.strip(), len(str_s.strip()))
                    #print(pred_s, prob)
                    if pred_s == real_s:
                        # exactly the same string
                        cl_good_equal += 1
                    else:
                        se_good = 0
                        if predictor.LABEL_LAST in pred_s:
                            n_examples_not_good_with_EOL += 1
                        #print(real_s, str_s.strip(), len(str_s.strip()))
                        #print(pred_s, prob)
                n_examples_equal += cl_good_equal
                n_sentences_good += se_good

            print(u"  Examples")
            print(u"    Correct (equ): %d/%d (%.3f)" % (n_examples_equal, n_examples, float(n_examples_equal) / n_examples))
            print(u"    Incorrect EOL: %d/%d (%.3f)" % (n_examples_not_good_with_EOL, n_examples, float(n_examples_not_good_with_EOL) / n_examples))
            print(u"  Sentences")
            print(u"    Correct:       %d/%d (%.3f)" % (n_sentences_good, n_sentences, float(n_sentences_good) / n_sentences))
            print(u"Evaluating pure labelling... done")
            print(u"")

        def evaluate_splitter(example_sentences, language, model_file_path):
            print(u"Evaluating splitter...")
            n_sentences = len(example_sentences)
            n_sentences_equal = 0
            n_sentences_same_length = 0
            n_sentences_different = 0
            for example_sentence in example_sentences:
                # TODO itertools...
                real_lines = []
                for _, _, tokens in example_sentence:
                    real_lines.append(u"".join([t.augmented_string for t in tokens]).strip())
                #print(real_lines)

                lines = [l for l in example_sentence]
                #features = [l[0] for l in lines]
                labels = [l[1] for l in lines]
                tokens = [l[2] for l in lines]
                #features = [item for sublist in features for item in sublist]
                labels = [item for sublist in labels for item in sublist]
                tokens = [item for sublist in tokens for item in sublist]

                splitter = CRFSplitter(language=language, model_file_path=model_file_path)
                ccs = splitter._split_sentence(TokenizedSentenceSpan(elements=tokens))
                predicted_lines = []
                for cc in ccs:
                    for line in cc.elements:
                        predicted_lines.append(line.clean_string)
                #print(predicted_lines)

                if predicted_lines == real_lines:
                    n_sentences_equal += 1
                elif len(predicted_lines) == len(real_lines):
                    n_sentences_same_length += 1
                    print(u"ER %d : " % (len(real_lines)) + u" | ".join(real_lines))
                    print(u"EP %d : " % (len(predicted_lines)) + u" | ".join(predicted_lines))
                    print(u"")
                else:
                    n_sentences_different += 1
                    print(u"DR %d : " % (len(real_lines)) + u" | ".join(real_lines))
                    print(u"DP %d : " % (len(predicted_lines)) + u" | ".join(predicted_lines))
                    print(u"")

            print(u"  Sentences split exactly:     %d/%d (%.3f)" % (n_sentences_equal, n_sentences, float(n_sentences_equal) / n_sentences))
            print(u"  Sentences split same length: %d/%d (%.3f)" % (n_sentences_same_length, n_sentences, float(n_sentences_same_length) / n_sentences))
            print(u"  Sentences split differently: %d/%d (%.3f)" % (n_sentences_different, n_sentences, float(n_sentences_different) / n_sentences))
            print(u"Evaluating splitter... done")
            print(u"")

        print(u"Loading data...")
        trainer = CRFTrainer(language=language, nlpwrapper=u"nltk")
        arg = [dump_file_path] if single else dump_file_path
        trainer.load_data(arg)
        #examples = trainer.train_data
        example_sentences = trainer.train_data
        print(u"Loading data... done")

        print(u"Testing...")

        # first, test pure labelling of lines
        if True:
            predictor = CRFPredictor(model_file_path)
            evaluate_pure_labelling(example_sentences, predictor)

        # then, mock the splitter behavior
        if not True:
            evaluate_splitter(example_sentences, language, model_file_path)

        print(u"Testing... done")

    ##########################################################################
    #
    # main script stats here
    #
    ##########################################################################
    if len(sys.argv) < 5:
        usage(1)

    command = sys.argv[1]
    single = u"--single" in sys.argv
    small_n = None
    for a in sys.argv:
        if a.startswith(u"--small="):
            small_n = int(a[len(u"--small="):])

    if command not in [u"dump", u"train", u"test"]:
        print(u"[ERRO] Unknown command '%s'" % command)
        usage(1)

    if command == u"dump":
        language = check_language(sys.argv[2])
        input_directory_path = check_dir(sys.argv[3])
        dump_file_path = sys.argv[4]
        command_dump(language, input_directory_path, dump_file_path, small_n)

    if command == u"train":
        language = check_language(sys.argv[2])
        dump_file_path = check_file(sys.argv[3])
        model_file_path = sys.argv[4]
        command_train(language, dump_file_path, model_file_path)

    if command == u"test":
        language = check_language(sys.argv[2])
        dump_file_path = check_file(sys.argv[3])
        model_file_path = check_file(sys.argv[4])
        command_test(language, dump_file_path, model_file_path, single)

    sys.exit(0)


if __name__ == "__main__":
    main()
