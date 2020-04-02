import generate_gt_from_txt_w
import generate_gt_from_xml_w
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
        xml_data_set = generate_gt_from_xml_w.get_gt(s)
        txt_data_set = generate_gt_from_txt_w.get_gt(s)

        xml_set = set([hashabledict(x) for x in xml_data_set])
        txt_set = set([hashabledict(x) for x in txt_data_set])

        print len(xml_set)
        print len(txt_set)
        print "Sym Diff (should be zero):", len(xml_set ^ txt_set)
