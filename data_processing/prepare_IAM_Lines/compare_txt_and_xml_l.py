import generate_gt_from_txt_l
import generate_gt_from_xml_l
import load_set

class hashabledict(dict):
  def __key(self):
    return tuple((k,self[k]) for k in sorted(self))
  def __hash__(self):
    return hash(self.__key())
  def __eq__(self, other):
    return self.__key() == other.__key()

if __name__ == "__main__":
    sets = load_set.load()

    for s in sets:
        xml_data_set = generate_gt_from_xml_l.get_gt(s)
        txt_data_set = generate_gt_from_txt_l.get_gt(s)

        xml_set = set([hashabledict(x) for x in xml_data_set])
        txt_set = set([hashabledict(x) for x in txt_data_set])
        print "Length of xml:", len(xml_set)
        print "Length of txt:", len(txt_set)
        print "Sym Diff (Not zero because of space and punctuation issue):", len(xml_set ^ txt_set)

        for x in xml_set:
            x['gt'] = "".join(x['gt'].split())

        for x in txt_set:
            x['gt'] = "".join(x['gt'].split())

        #need to reset the hashabledict
        xml_set = set([hashabledict(x) for x in xml_set])
        txt_set = set([hashabledict(x) for x in txt_set])

        print "Sym Diff (No whitespace, should be zero):", len(xml_set ^ txt_set)
        print ""


        # re.sub("([^ ])'" , "\g<1> '", text1)
