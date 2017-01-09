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
import io
import re
import os
import youtube_dl
from lxml import etree

from lachesis.downloaders import NotDownloadedError
import lachesis.globalfunctions as gf


class YDLogger(object):
    """
    TBW
    """
    def debug(self, msg):
        #print(u"[DEBU] %s" % msg)
        pass

    def warning(self, msg):
        #print(u"[WARN] %s" % msg)
        pass

    def error(self, msg):
        #print(u"[ERRO] %s" % msg)
        pass


class YouTubeDownloader(object):

    CODE = u"youtube"

    OPTION_AUTO = u"auto"

    @classmethod
    def can_download(cls, url, downloader):
        """
        Determine if this class can download the given URL.
        """
        return (
            (downloader == cls.CODE) or
            (u"youtube.com" in url) or
            (u"youtu.be" in url) or
            (len(url) == 11)
        )

    @classmethod
    def get_data(cls, url, language, options):
        auto = options.get(cls.OPTION_AUTO, False)
        handler, tmp = gf.tmp_file()
        if os.path.exists(tmp):
            os.remove(tmp)
        tmp = gf.to_unicode_string(tmp)
        out = u"%s.%s.ttml" % (tmp, language)
        ydl_options = {
            "outtmpl": tmp,
            "subtitlesformat": u"ttml",
            "subtitleslangs": [language],
            "writesubtitles": not auto,
            "writeautomaticsub": auto,
            "skip_download": True,
            "logger": YDLogger(),
        }
        try:
            with youtube_dl.YoutubeDL(ydl_options) as ydl:
                ydl.download([url])
        except Exception as e:
            raise NotDownloadedError
        if not os.path.exists(out):
            raise NotDownloadedError
        with io.open(out, "r", encoding="utf-8") as out_file:
            data = out_file.read()
        gf.delete_file(handler, tmp)
        gf.delete_file(None, out)
        return data


    @classmethod
    def parse_raw_data(cls, raw):
        """
        Parse the given raw data string, and return a list of tuples:

            [
                (begin_1, end_1, [line_1_1, line_1_2, ...]),
                (begin_2, end_2, [line_2_1]),
                ...
            ]

        """
        # constants
        PLACEHOLDER_BR = u" ||| "
        PLACEHOLDER_NO_TEXT = u"()"
        PATTERN_SPAN_OPEN = re.compile(r"<span[^>]*>")
        PATTERN_SPAN_CLOSE = re.compile(r"</span>")
        PATTERN_BR = re.compile(r"<br[ ]*/>")
        PATTERN_SPACES = re.compile(r"\s+")
        TTML_NS = "{http://www.w3.org/ns/ttml}"
        TTML_P = TTML_NS + "p"
        TTML_BEGIN = "begin"
        TTML_END = "end"

        # remove spans
        s = raw
        s = re.sub(PATTERN_SPAN_OPEN, u"", s)
        s = re.sub(PATTERN_SPAN_CLOSE, u"", s)
        # replace br with placeholder
        s = re.sub(PATTERN_BR, PLACEHOLDER_BR, s)
        # remove duplicated spaces
        s = re.sub(PATTERN_SPACES, u" ", s).strip()

        # encode to utf-8 as required by lxml
        if gf.is_unicode(raw):
            s = s.encode("utf-8")

        # parse fragments
        frags = []
        root = etree.fromstring(s)
        for elem in root.iter(TTML_P):
            begin = elem.get(TTML_BEGIN).strip()
            end = elem.get(TTML_END).strip()
            text = elem.text
            # text missing
            if text is None:
                text = u""
            # strip leading/trailing spaces
            text = text.strip()
            # if no text is available, replace it with ()
            if text == u"":
                text = PLACEHOLDER_NO_TEXT
            # split lines if the <br/> is present
            pieces = [t.strip() for t in text.split(PLACEHOLDER_BR)]
            # make sure we return unicode strings
            pieces = [gf.to_unicode_string(t) for t in pieces if len(t) > 0]
            frags.append((begin, end, pieces))
        return frags
