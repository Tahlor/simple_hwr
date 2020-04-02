from hwr_utils.hwr_utils import pickle_it
import os
from xml.dom import minidom
from collections import defaultdict

xml_obj = None
def get_writers_offline(xml_obj, path):
    parent, child = os.path.split(path)
    return xml_obj.getElementsByTagName('form')[0].getAttribute("writer-id"), child[:-4][0:7]

def get_writers_online(xml_obj, path):
    parent, child = os.path.split(path)
    parent, child = os.path.split(parent)
    return xml_obj.getElementsByTagName('Form')[0].getAttribute("writerID"), child[0:7]

def loop_through_xmls(path, process_func=get_writers_offline):
    global xml_obj

    output_dict = defaultdict(list)

    for d,s,fs in os.walk(path):
        for fn in fs:
            curr_path = os.path.join(d, fn)
            try:
                xml_obj = minidom.parse(curr_path)
                key, value = process_func(xml_obj, curr_path)
                output_dict[key].append(value)
            except:
                pass

    return output_dict
    #with open(curr_path, "r") as f:
    #my_func

if __name__=="__main__":
    my_dict = {"online":{"folder":"prepare_online_data", "subfolder": "original-xml-all", "func":get_writers_online}, "offline":{"folder":"prepare_IAM_Lines", "subfolder": "xml", "func":get_writers_offline}}
    for variant in ["offline", "online"]:
        print("Processing {} writers".format(variant))
        folder = my_dict[variant]["folder"]
        root = os.path.join("./data", folder)
        xml_path = os.path.join(root, my_dict[variant]["subfolder"])
        output = loop_through_xmls(xml_path, process_func=my_dict[variant]["func"])
        pickle_it(output, os.path.join(root, "writer_IDs.pickle"))
        #print(output)