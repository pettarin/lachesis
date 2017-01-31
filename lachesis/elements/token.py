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

from lachesis.language import Language
import lachesis.globalfunctions as gf


class Token(object):
    """
    A Token is a word or punctuation in a Document.
    """

    def __init__(
        self,
        raw,
        upos_tag=None,
        chunk_tag=None,
        pnp_tag=None,
        lemma=None,
        trailing_whitespace=False
    ):
        self.raw = raw
        self.upos_tag = upos_tag
        self.chunk_tag = chunk_tag
        self.pnp_tag = pnp_tag
        self.lemma = lemma
        self.trailing_whitespace = trailing_whitespace
        self.special = False

    def __str__(self):
        return self.augmented_string

    @property
    def is_special(self):
        """
        Return ``True`` if this token is special
        (e.g., a token introduced for analysis purposes
        that must be ignored otherwise).
        """
        return self.special

    @property
    def is_regular(self):
        """
        Return ``True`` if this token is regular
        (i.e., not special).
        """
        return not self.is_special

    @property
    def raw_string(self):
        """
        Return the raw string of the token,
        NOT including the trailing whitespace, if any.
        """
        return self.raw

    @property
    def augmented_string(self):
        """
        Return a string representation of the token,
        including a trailing whitespace
        (if the corresponding flag is set).
        """
        if self.trailing_whitespace:
            return self.raw + u" "
        return self.raw

    @property
    def tagged_string(self):
        """
        Return a tagged representation of the token:

        ``raw/upos/ws``

        where ``ws`` is ``Y`` (yes), ``N`` (no),
        or ``S`` (special).
        """
        if self.is_special:
            ws = u"S"
        elif self.trailing_whitespace:
            ws = u"Y"
        else:
            ws = u"N"
        return u"%s/%s/%s" % (self.raw, self.upos_tag, ws)


class SpecialToken(Token):

    RAW = u"lachesisspecialtoken"
    UPOS_TAG = u"LAC"

    def __init__(self):
        super(SpecialToken, self).__init__(
            self.RAW,
            upos_tag=self.UPOS_TAG
        )
        self.trailing_whitespace = True
        self.special = True


class EndOfSentenceToken(SpecialToken):

    RAW = u"lachesisendofsentencetoken"
    UPOS_TAG = u"XXX"


class EndOfCCToken(SpecialToken):

    RAW = u"lachesisendofcctoken"
    UPOS_TAG = u"YYY"


class EndOfLineToken(SpecialToken):

    RAW = u"lachesisendoflinetoken"
    UPOS_TAG = u"ZZZ"
