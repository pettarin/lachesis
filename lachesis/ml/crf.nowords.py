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
import pickle
import io
import os
import re
import sys

from lachesis.downloaders import Downloader
from lachesis.elements import Span
from lachesis.language import Language
from lachesis.nlpwrappers import NLPEngine
from lachesis.nlpwrappers import UniversalPOSTags
import lachesis.globalfunctions as gf


LABEL_FIRST = u"F"
""" Label for a token that is the begin of a line """

LABEL_MIDDLE = u"_"
""" Label for a token that is not the last of a line """

LABEL_LAST = u"L"
""" Label for a token that is the last of a line """

SPECIAL_POS = [
    UniversalPOSTags.INTJ,
    UniversalPOSTags.NUM,
    UniversalPOSTags.PUNCT,
    UniversalPOSTags.SYM,
    UniversalPOSTags.X
]
""" Some special POS """


def sentence_to_features(
    sentence,
    look_behind=5,
    look_ahead=20,
    max_chars_per_line=42,
    debug=False
):
    """
    Convert a sequence of tokens into a sequence of features,
    that is, a list of dicts, each dict containing
    the features associated to the corresponding token
    in the input sequence.
    """

    def from_previous_pos(idx, poses, clens, target_poses):
        """ Get the distance, in chars, from the previous (say) NOUN """
        if idx == 0:
            return 0
        for i in range(idx - 1, -1, -1):
            if poses[i] in target_poses:
                return clens[idx] - clens[i]
        return 0

    def subarray(arr, idx, from_offset, to_offset):
        if idx + from_offset < 0:
            return []
        if idx + to_offset >= len(arr):
            return []
        return arr[idx + from_offset:idx + to_offset + 1]

    def add_feature(feat, label, arr, idx, from_offset, to_offset):
        if from_offset == to_offset:
            label = label % (from_offset)
        else:
            label = label % (from_offset, to_offset)
        sub = subarray(arr, idx, from_offset, to_offset)
        if len(sub) == 1:
            feat[label] = sub[0]
        elif len(sub) > 0:
            if gf.is_unicode(sub[0]):
                value = u"-".join(sub)
            elif isinstance(sub[0], int):
                value = sub[-1] - sub[0]
            else:
                value = sub
            feat[label] = value

    # list of features to be returned, each is a dict
    feature_sequence = []
    # we only work only on regular tokens
    tokens = sentence.regular_tokens
    # raw string of the word
    words = [t.raw for t in tokens]
    # POS of the word
    poss = [t.upos_tag for t in tokens]
    # bool, True if word has trailing whitespace
    wses = [1 if t.trailing_whitespace else 0 for t in tokens]

    n = len(tokens)

    # length of the entire sentence once cleaned
    m = len(sentence.clean_string)

    # lens[i] = length of the i-th (raw) word
    # lens = [len(w) for w in words]

    # plens[i] = length of the prefix up to i-th word,
    #            including ws
    plens = sentence.prefix_lengths

    # rlens[i] = length of the suffix starting at the (i+1)-th word,
    #            including ws
    rlens = [m - p for p in plens]

    for idx in range(n):
        feat = {
            # "word": words[idx],
            "pos": poss[idx],
            "ws": wses[idx],
            # "len": lens[idx],
        }
        if idx == 0:
            feat["bos"] = True
        elif idx == n - 1:
            feat["eos"] = True
        else:
            for offset in range(1, 1 + look_behind):
                try:
                    if plens[idx] - plens[idx - offset] <= max_chars_per_line:
                        # add_feature(feat, "ws_%d", wses, idx, -offset, -offset)
                        add_feature(feat, "pos_%d", poss, idx, -offset, -offset)
                        add_feature(feat, "pos_%d_%d", poss, idx, -offset, 0)
                        add_feature(feat, "plen_%d_%d", plens, idx, -offset, 0)
                except:
                    pass

            for offset in range(1, 1 + look_ahead):
                try:
                    if plens[idx + offset + 1] - plens[idx] <= max_chars_per_line:
                        # add_feature(feat, "ws_%d", wses, idx, +offset, +offset)
                        add_feature(feat, "pos_%d", poss, idx, +offset, +offset)
                        add_feature(feat, "pos_%d_%d", poss, idx, 0, +offset)
                        add_feature(feat, "plen_%d_%d", plens, idx, 0, +offset)
                        add_feature(feat, "rlen_%d_%d", rlens, idx, 0, +offset)
                except:
                    pass

            for offset in range(1, 1 + min(look_behind, look_ahead)):
                try:
                    if plens[idx + offset] - plens[idx - offset] <= max_chars_per_line:
                        add_feature(feat, "pos_%d_%d", poss, idx, -offset, +offset)
                        add_feature(feat, "plen_%d_%d", plens, idx, -offset, +offset)
                except:
                    pass

        for k in feat.keys():
            if k.startswith("plen_"):
                feat[k] = feat[k] > max_chars_per_line
            elif k.startswith("rlen_"):
                feat[k] = feat[k] <= max_chars_per_line

        # print(feat)
        feature_sequence.append(feat)

    return feature_sequence


def sentence_to_labels(sentence):
    labels = [[LABEL_MIDDLE for t in l.tokens] for l in sentence.lines]
    for line in labels:
        line[-1] = LABEL_LAST
    return reduce(lambda x, y: x + y, labels, [])


class CRFTrainer(object):
    """
    TBW
    """

    PARAMETERS = {
        # algorithms: lbfgs, l2sgd, ap, pa, arow
        "algorithm": "arow",
        "max_iterations": 5000,
        # "feature.possible_states": True,
        # "feature.possible_transitions": True,
    }
    """ Parameters for the trainer from pycrfsuite """

    VERBOSE = False
    """ Verbosity of the trainer """

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
        #
        # NOTE in theory we could account for things like the position
        #      of a token w.r.t. the whole document or the sentence it belongs
        #      however here we make an assumption of independence
        #      of the split point from those facts
        #
        examples = []
        for ifp in input_file_paths:
            print(u".")
            if os.path.isfile(ifp):
                doc = Downloader.read_closed_captions(
                    ifp,
                    {u"downloader": u"youtube"}
                )
                self.nlpe.analyze(doc, wrapper=self.nlpwrapper)
                for sentence in doc.sentences:
                    features = sentence_to_features(sentence)
                    labels = sentence_to_labels(sentence)
                    example = (sentence, features, labels)
                    examples.append(example)
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
        self.trainer = pycrfsuite.Trainer(algorithm=self.parameters["algorithm"], verbose=self.verbose)

        # append training data
        for sentence, features, labels in self.train_data:
            # each example is a triplet (sentence, features, labels)
            # where:
            # - sentence is a TokenizedSentenceSpan
            # - features is a list of dicts
            # - labels is a list of labels (e.g., LABEL_MIDDLE or LABEL_LAST)
            self.trainer.append(features, labels)

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

    LABEL_FIRST = LABEL_FIRST

    LABEL_MIDDLE = LABEL_MIDDLE

    LABEL_LAST = LABEL_LAST

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
            features = sentence_to_features(obj)
        else:
            raise TypeError(u"The obj should be either a Span (sentence) object or a list of features (dict) objects.")
        predicted_labels = [gf.to_unicode_string(l) for l in self.tagger.tag(features)]
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

    def command_test(language, dump_file_path, model_file_path, single, pure):
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

        def pg(string):
            print("%s%s%s" % (ANSI_OK, string, ANSI_END))

        def pe(string):
            print("%s%s%s" % (ANSI_ERROR, string, ANSI_END))

        def check_prediction(predictor, sentence, features, labels):
            """
            Call the tagger and check the prediction
            against the true value (i.e., the true label).
            """
            labels_pred, probability = predictor.predict(features)
            real_l = u"".join(labels)
            real_s = sentence.marked_string(eol=u"|")
            pred_l = u"".join(labels_pred)
            pred_s = u""
            for t, l in zip(sentence.regular_tokens, labels_pred):
                pred_s += t.augmented_string
                if l == LABEL_LAST:
                    pred_s += u" | "
            pred_s = pred_s.strip()
            pred_s = re.sub(r" [ ]*", u" ", pred_s)
            return (real_l, pred_l, real_s, pred_s, probability)

        def evaluate_pure_labelling(examples, predictor):
            """
            Evaluate the pure labelling of lines, that is,
            get the label string for a line
            (knowing that it is the whole line),
            and confront it with the ground truth label string
            (which, in particular, has form: "___...___E").
            """
            print(u"Evaluating pure labelling...")
            n_sentences = len(examples)
            n_sentences_good = 0
            n_symbols = 0
            n_symbols_good = 0
            for sentence, features, labels in examples:
                real_l, pred_l, real_s, pred_s, prob = check_prediction(predictor, sentence, features, labels)
                n_sentences_good += 1 if (pred_l == real_l) else 0
                n_symbols += len(real_l)
                n_symbols_good += sum([1 if p == r else 0 for p, r in zip(pred_l, real_l)])
                if True:
                    pf = pg if (pred_l == real_l) else pe
                    pf(u"R: " + real_s)
                    pf(u"R: " + u"".join(real_l))
                    pf(u"P: " + u"".join(pred_l))
                    pf(u"R: " + pred_s + u" %.3f" % prob)
                    pf(u"")

            print(u"  Sentences")
            print(u"    Correct:       %d/%d (%.3f)" % (n_sentences_good, n_sentences, float(n_sentences_good) / n_sentences))
            print(u"  Symbols")
            print(u"    Correct:       %d/%d (%.3f)" % (n_symbols_good, n_symbols, float(n_symbols_good) / n_symbols))
            print(u"Evaluating pure labelling... done")
            print(u"")

        def evaluate_splitter(examples, language, model_file_path):
            print(u"Evaluating splitter...")
            n_sentences = len(examples)
            n_lines = 0
            n_sentences_good = 0
            n_lines_good = 0
            splitter = CRFSplitter(language=language, model_file_path=model_file_path)
            for sentence, features, labels in examples:
                ccs = splitter._split_sentence(sentence)
                splitter_lines = reduce(lambda x, y: x + y, [cc.lines for cc in ccs], [])
                real_lines = sentence.lines
                n_lines += len(real_lines)
                # print(u"SENT: " + sentence.marked_string(eol=u"|"))
                # for l in real_lines:
                #     print(u"  LINE: " + l.augmented_string)
                all_good = True
                if len(real_lines) == len(splitter_lines):
                    for rl, sl in zip(real_lines, splitter_lines):
                        rla = rl.clean_string
                        sla = sl.clean_string
                        if rla != sla:
                            all_good = False
                        else:
                            n_lines_good += 1
                else:
                    all_good = False
                n_sentences_good += 1 if all_good else 0

                rs = sentence.marked_string(eol=u"|")
                ss = u" | ".join([s.clean_string for s in splitter_lines])
                pf = pg if rs == ss else pe
                pf(u"R: " + rs)
                pf(u"P: " + ss)
                pf(u"")

            print(u"  Lines")
            # print(u"    Correct:       %d/%d (%.3f)" % (n_lines_good, n_lines, float(n_lines_good) / n_lines))
            # print(u"  Sentences")
            print(u"    Correct:       %d/%d (%.3f)" % (n_sentences_good, n_sentences, float(n_sentences_good) / n_sentences))
            print(u"Evaluating splitter... done")

        print(u"Loading data...")
        trainer = CRFTrainer(language=language, nlpwrapper=u"nltk")
        arg = [dump_file_path] if single else dump_file_path
        trainer.load_data(arg)
        example_sentences = trainer.train_data
        print(u"Loading data... done")

        print(u"Testing...")

        if pure:
            # test pure labelling of lines
            predictor = CRFPredictor(model_file_path)
            evaluate_pure_labelling(example_sentences, predictor)
        else:
            # mock the splitter behavior
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
    pure = u"--pure" in sys.argv
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
        command_test(language, dump_file_path, model_file_path, single, pure)

    sys.exit(0)


if __name__ == "__main__":
    main()
