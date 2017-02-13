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

from lachesis.language import Language
from lachesis.nlpwrappers.base import BaseWrapper


class PolyglotWrapper(BaseWrapper):
    """
    TBW
    """

    CODE = u"polyglot"

    MODEL_FILES_DIRECTORY_PATH = os.path.join(
        os.path.expanduser("~/"),
        "lachesis_data",
        "polyglot_data"
    )

    LANGUAGE_TO_POLYGLOT_CODE = {
        Language.BULGARIAN: u"bg",
        Language.CZECH: u"cs",
        Language.DANISH: u"da",
        Language.DUTCH: u"nl",
        Language.ENGLISH: u"en",
        Language.FINNISH: u"fi",
        Language.FRENCH: u"fr",
        Language.GERMAN: u"de",
        Language.HUNGARIAN: u"hu",
        Language.INDONESIAN: u"id",
        Language.IRISH: u"ga",
        Language.ITALIAN: u"it",
        Language.PORTUGUESE: u"pt",
        Language.SLOVENIAN: u"sl",
        Language.SPANISH: u"es",
        Language.SWEDISH: u"sv",
    }

    LANGUAGES = LANGUAGE_TO_POLYGLOT_CODE.keys()

    def __init__(self, language):
        super(PolyglotWrapper, self).__init__(language)
        self.polyglot_language = self.LANGUAGE_TO_POLYGLOT_CODE[self.language]
        # TODO does not seem to work?
        # import polyglot
        # polyglot.data_path = self.MODEL_FILES_DIRECTORY_PATH

    def _analyze(self, doc_string):
        sentences = []
        from polyglot.text import Text
        lib_sentences = Text(doc_string, hint_language_code=self.polyglot_language).sentences
        for lib_sentence in lib_sentences:
            sentence_tokens = []
            for lib_token, lib_upos_tag in lib_sentence.pos_tags:
                token = self._create_token(lib_token, lib_upos_tag)
                sentence_tokens.append(token)
            sentences.append(sentence_tokens)
        return sentences
