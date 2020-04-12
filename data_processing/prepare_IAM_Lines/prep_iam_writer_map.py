import sys


from html.parser import HTMLParser
import xml.etree.ElementTree
from os import listdir
from os.path import isfile, join
import re
import json
import numpy as np

def get_mapping(xml_folder):
    onlyfiles = [join(xml_folder, f) for f in listdir(xml_folder) if isfile(join(xml_folder, f))]

    all_line_gts = {}
    all_word_gts = {}
    mapping = {}
    for f in onlyfiles:
        form_id, writer_id, avg_line, avg_full, line_gts, word_gts = get_key_value(f)
        mapping[form_id] = writer_id, avg_line, avg_full
        all_line_gts.update(line_gts)
        all_word_gts.update(word_gts)

    return mapping, all_line_gts, all_word_gts

def get_namespace(element):
    m = re.match('\{.*\}', element.tag)
    return m.group(0) if m else ''

def get_key_value(xml_file):
    root = xml.etree.ElementTree.parse(xml_file).getroot()
    namespace = get_namespace(root)

    handwritten_part = root.find('handwritten-part')
    lbys = []
    dys = []
    line_gts = {}
    word_gts = {}

    for lines in handwritten_part:

        in_error = False
        if lines.attrib['segmentation'] == 'err':
            in_error = True

        lby = float(lines.attrib['lby'])
        uby = float(lines.attrib['uby'])

        asy = float(lines.attrib['asy'])
        dsy = float(lines.attrib['dsy'])

        dys.append(dsy - asy)
        lbys.append(lby - uby)

        h = HTMLParser()
        line_text = h.unescape(lines.attrib['text'])

        line_gts[lines.attrib['id']] = {
            "gt": line_text,
            "err": in_error
        }
        for word in lines.findall('word'):
            word_text = h.unescape(word.attrib['text'])
            word_gts[word.attrib['id']] = {
                "gt": word_text,
                "err": in_error
            }

    return root.attrib['id'], root.attrib['writer-id'], np.median(lbys), np.median(dys), line_gts, word_gts


if __name__ == "__main__":
    xml_folder = sys.argv[1]
    output_file = sys.argv[2]

    mapping = get_mapping(xml_folder)

    with open(output_file, "w") as f:
        json.dump(mapping, f)
