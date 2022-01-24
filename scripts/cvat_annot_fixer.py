import sys

from lxml import objectify, etree


def read_annot(path, front_list):
    root = objectify.parse(path).getroot()
    tracks = [child for child in root.iterchildren()][2:]

    for track in tracks:
        track_id = int(track.attrib['id']) + 1
        frames = track.getchildren()
        for frame in frames:
            if track_id in front_list:
                frame.attrib['z_order'] = "1"

    obj_xml = etree.tostring(root, encoding="utf-8", pretty_print=True, xml_declaration=True)

    with open("0-953.xml", "wb") as xml_writer:
        xml_writer.write(obj_xml)


if __name__ == "__main__":
    front_list = [5,18, 28, 29, 33, 34]
    path = sys.argv[1]
    read_annot(path, front_list)
