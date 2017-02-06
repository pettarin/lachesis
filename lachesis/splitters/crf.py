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
import os

from lachesis.elements import CCLineSpan
from lachesis.elements import CCSpan
from lachesis.language import Language
from lachesis.ml import CRFPredictor
from lachesis.splitters.base import BaseSplitter


class CRFSplitter(BaseSplitter):
    """
    TBW
    """

    CODE = u"crf"

    CHECK_LANGUAGE = False

    MODEL_FILES_DIRECTORY_PATH = os.path.join(
        os.path.expanduser("~/"),
        "lachesis_data",
        "crf_data"
    )

    LANGUAGE_TO_MODEL_FILE = {
        Language.ENGLISH: [(u"eng.42.5.nltk.crfsuite", 42, 5, u"nltk")],
        Language.ITALIAN: [(u"ita.42.5.nltk.crfsuite", 42, 5, u"nltk")],
    }

    LANGUAGES = LANGUAGE_TO_MODEL_FILE.keys()

    def __init__(self, language, max_chars_per_line=BaseSplitter.MAX_CHARS_PER_LINE, max_num_lines=BaseSplitter.MAX_NUM_LINES, model_file_path=None):
        super(CRFSplitter, self).__init__(language, max_chars_per_line, max_num_lines)
        self.model_file_path = model_file_path
        if self.model_file_path is None:
            # first, check the language
            self._check_language()
            # then, find the path of the model file
            # best matching the splitter parameters
            # TODO currently serving only the first model
            mfp = ((self.LANGUAGE_TO_MODEL_FILE[self.language])[0])[0]
            self.model_file_path = os.path.join(self.MODEL_FILES_DIRECTORY_PATH, mfp)
        if not os.path.isfile(self.model_file_path):
            raise ValueError(u"Unable to load CRF model '%s'. Please download the file in that path, or provide your own path with the model_file_path parameter." % self.model_file_path)

    def _split_sentence(self, sentence_span):

        def _find_lines(tokens, labels):
            labels[-1] = CRFPredictor.LABEL_LAST
            lines = []
            cl = []
            for t, l in zip(tokens, labels):
                cl.append(t)
                if l == CRFPredictor.LABEL_LAST:
                    lines.append(cl)
                    cl = []
            return [CCLineSpan(elements=l) for l in lines]

        def _merge(lines):
            merged = []
            n = len(lines)
            i = 0
            while i < n:
                solved = False
                if i + 1 < n:
                    united = CCLineSpan(elements=(lines[i].elements + lines[i + 1].elements))
                    if len(united.clean_string) <= self.max_chars_per_line:
                        merged.append(united)
                        i += 1
                        solved = True
                if (not solved) and (i > 0):
                    united = CCLineSpan(elements=(lines[i - 1].elements + lines[i].elements))
                    if len(united.clean_string) <= self.max_chars_per_line:
                        merged[-1] = united
                        solved = True
                if not solved:
                    merged.append(lines[i])
                i += 1
            return merged

        def _lines_to_objects(lines):
            n = len(lines)
            pairs = [
                [lines[i], lines[i + 1]] if i + 1 < n else [lines[i]]
                for i in range(0, n, 2)
            ]
            return [CCSpan(elements=p) for p in pairs]

        # check for e.g. "(applause)" or similar OTHER fragments
        if self._is_cc_other(sentence_span):
            line = CCLineSpan(elements=sentence_span.elements)
            return [CCSpan(elements=[line])]

        # otherwise, process it
        tokens = [t for t in sentence_span.tokens if t.is_regular]
        clean_sentence_span = CCLineSpan(elements=tokens)

        # if the tokens fit into a single line,
        # create a CC with one line and return it
        if len(clean_sentence_span.clean_string) <= self.max_chars_per_line:
            return [CCSpan(elements=[clean_sentence_span])]

        # we actually need to create more than one line
        # load the CRF model
        ccs = []
        predictor = CRFPredictor(self.model_file_path)
        pred_labels, pred_probability = predictor.predict(clean_sentence_span)

        lines = _find_lines(tokens, pred_labels)
        lines = _merge(lines)
        ccs = _lines_to_objects(lines)

        # for cc in ccs:
        #     print(cc.marked_string(eol=u" | "))

        return ccs
