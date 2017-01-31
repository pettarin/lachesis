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
import re

from lachesis.elements.token import EndOfCCToken
from lachesis.elements.token import EndOfLineToken
from lachesis.elements.token import EndOfSentenceToken
from lachesis.language import Language
import lachesis.globalfunctions as gf


def clean(string, eoc=u"", eol=u"", eos=u""):
    s = string
    s = s.replace(EndOfCCToken.RAW, eoc)
    s = s.replace(EndOfLineToken.RAW, eol)
    s = s.replace(EndOfSentenceToken.RAW, eos)
    s = s.replace(u"\n", u" ")
    s = re.sub(r" [ ]*", u" ", s)
    s = s.strip()
    return s


class Span(object):
    """
    A Span corresponds to an arbitrary sublist of Tokens of a Document.
    Its elements might be Token objects or other (nested) Span objects.
    """

    def __init__(self, raw=None, elements=None):
        self.raw = raw
        self.elements = [] if elements is None else elements

    def append(self, obj):
        self.elements.append(obj)

    def extend(self, lst):
        self.elements.extend(lst)

    def __str__(self):
        return self.clean_string

    @property
    def clean_string(self):
        return clean(self.flat_string)

    def marked_string(self, eoc=u"", eol=u"", eos=u""):
        return clean(self.augmented_string, eoc=eoc, eol=eol, eos=eos)


class TokenizedTextSpan(Span):

    @property
    def sentences(self):
        return self.elements

    @property
    def raw_string(self):
        return u"\n".join([s.raw_string for s in self.sentences])

    @property
    def augmented_string(self):
        return u"\n".join([s.augmented_string for s in self.sentences])

    @property
    def flat_string(self):
        return self.augmented_string


class TokenizedSentenceSpan(Span):

    @property
    def tokens(self):
        return self.elements

    @property
    def raw_string(self):
        return u" ".join([t.raw_string for t in self.tokens])

    @property
    def augmented_string(self):
        return u"".join([t.augmented_string for t in self.tokens])

    @property
    def flat_string(self):
        return self.augmented_string


class TextSpan(Span):
    """
    A Span holding either a raw string, or a list of strings,
    each supposed to be a sentence.
    """

    @property
    def sentences(self):
        return self.elements

    @property
    def raw_string(self):
        if self.raw is not None:
            return self.raw
        return u"\n".join([s.raw_string for s in self.sentences])

    @property
    def augmented_string(self):
        if self.raw is not None:
            return self.raw
        return u"\n".join([s.augmented_string for s in self.sentences])

    @property
    def flat_string(self):
        if self.raw is not None:
            return self.raw
        return u" ".join([s.flat_string for s in self.sentences])


class SentenceSpan(Span):
    """
    A Span holding a raw string without new line characters,
    representing a sentence.
    """

    def __init__(self, raw):
        self.raw = clean(raw)
        self.elements = None

    @property
    def raw_string(self):
        return self.raw

    @property
    def augmented_string(self):
        return u"%s %s" % (self.raw, EndOfSentenceToken.RAW)

    @property
    def flat_string(self):
        return u"%s %s" % (self.raw, EndOfSentenceToken.RAW)


class CCListSpan(Span):
    """
    A Span holding a list of CCs.
    """

    @property
    def ccs(self):
        return self.elements

    @property
    def raw_string(self):
        return u"\n\n".join([cc.raw_string for cc in self.ccs])

    @property
    def augmented_string(self):
        return u"\n\n".join([cc.augmented_string for cc in self.ccs])

    @property
    def flat_string(self):
        return u" ".join([cc.flat_string for cc in self.ccs])


class CCSpan(Span):
    """
    A Span holding a CC, that is, a list of CC line objects.
    Optionally, it can have an identifier and a time interval
    associated to it.
    """

    def __init__(self, raw=None, elements=[], identifier=None, time_interval=None):
        super(CCSpan, self).__init__(raw=raw, elements=elements)
        self.identifier = identifier
        self.time_interval = time_interval

    @property
    def lines(self):
        return self.elements

    @property
    def raw_string(self):
        return u"\n".join([l.raw_string for l in self.lines])

    @property
    def augmented_string(self):
        return u"%s %s" % (u"\n".join([l.augmented_string for l in self.lines]), EndOfCCToken.RAW)

    @property
    def flat_string(self):
        return u"%s %s" % (u" ".join([l.augmented_string for l in self.lines]), EndOfCCToken.RAW)


class CCLineSpan(Span):
    """
    A Span representing a CC line, that is,
    a string without new lines in it.
    """

    def __init__(self, raw=None, elements=None):
        if raw is None:
            self.raw = None
        else:
            self.raw = clean(raw)
        self.elements = [] if elements is None else elements

    @property
    def tokens(self):
        return self.elements

    @property
    def line(self):
        return self.raw_string

    @property
    def raw_string(self):
        if self.raw is not None:
            return self.raw
        return u"".join([t.augmented_string for t in self.tokens])

    @property
    def augmented_string(self):
        return u"%s %s" % (self.raw_string, EndOfLineToken.RAW)

    @property
    def flat_string(self):
        return u"%s %s" % (self.raw_string, EndOfLineToken.RAW)
